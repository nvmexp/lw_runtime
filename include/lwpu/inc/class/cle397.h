/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2010 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */
#ifndef _cl_e397_h_
#define _cl_e397_h_

#include "lwtypes.h"
#define LW_E3_THREED                            (0xE397)
#define LWE397_ADDRBITS_PER_UNIT                8
#define LWE397_ILWALID_CHANNEL                  15
#define LWE397_CTL_INCR_SYNCPT_NB_CONDS         4

// Register LWE397_CTL_INCR_SYNCPT_0  
#define LWE397_CTL_INCR_SYNCPT_0                                  (0x0)
#define LWE397_CTL_INCR_SYNCPT_0_COND                             15:8
#define LWE397_CTL_INCR_SYNCPT_0_COND_IMMEDIATE                   (0)
#define LWE397_CTL_INCR_SYNCPT_0_COND_OP_DONE                     (1)
#define LWE397_CTL_INCR_SYNCPT_0_COND_RD_DONE                     (2)
#define LWE397_CTL_INCR_SYNCPT_0_COND_REG_WR_SAFE                 (3)
#define LWE397_CTL_INCR_SYNCPT_0_COND_COND_4                      (4)
#define LWE397_CTL_INCR_SYNCPT_0_COND_COND_5                      (5)
#define LWE397_CTL_INCR_SYNCPT_0_COND_COND_6                      (6)
#define LWE397_CTL_INCR_SYNCPT_0_COND_COND_7                      (7)
#define LWE397_CTL_INCR_SYNCPT_0_COND_COND_8                      (8)
#define LWE397_CTL_INCR_SYNCPT_0_COND_COND_9                      (9)
#define LWE397_CTL_INCR_SYNCPT_0_COND_COND_10                     (10)
#define LWE397_CTL_INCR_SYNCPT_0_COND_COND_11                     (11)
#define LWE397_CTL_INCR_SYNCPT_0_COND_COND_12                     (12)
#define LWE397_CTL_INCR_SYNCPT_0_COND_COND_13                     (13)
#define LWE397_CTL_INCR_SYNCPT_0_COND_COND_14                     (14)
#define LWE397_CTL_INCR_SYNCPT_0_COND_COND_15                     (15)
#define LWE397_CTL_INCR_SYNCPT_0_INDX                             7:0


// Register LWE397_CTL_INCR_SYNCPT_CNTRL_0  
#define LWE397_CTL_INCR_SYNCPT_CNTRL_0                            (0x1)
#define LWE397_CTL_INCR_SYNCPT_CNTRL_0_INCR_SYNCPT_NO_STALL       8:8
#define LWE397_CTL_INCR_SYNCPT_CNTRL_0_INCR_SYNCPT_SOFT_RESET     0:0


// Register LWE397_CTL_INCR_SYNCPT_ERROR_0  
#define LWE397_CTL_INCR_SYNCPT_ERROR_0                            (0x2)
#define LWE397_CTL_INCR_SYNCPT_ERROR_0_COND_STATUS                31:0


// Register LWE397_CTL_INTSTATUS_0  
#define LWE397_CTL_INTSTATUS_0                                    (0x8)
#define LWE397_CTL_INTSTATUS_0_CTXSW_INT                          0:0


// Register LWE397_CTL_INTENABLE_0  
#define LWE397_CTL_INTENABLE_0                                    (0x9)
#define LWE397_CTL_INTENABLE_0_CTXSW_INT                          0:0
#define LWE397_CTL_INTENABLE_0_CTXSW_INT_DISABLE                  (0)
#define LWE397_CTL_INTENABLE_0_CTXSW_INT_ENABLE                   (1)


// Register LWE397_CTL_CTXSW_0  
#define LWE397_CTL_CTXSW_0                                        (0xa)
#define LWE397_CTL_CTXSW_0_LWRR_CLASS                             9:0
#define LWE397_CTL_CTXSW_0_AUTO_ACK                               11:11
#define LWE397_CTL_CTXSW_0_AUTO_ACK_MANUAL                        (0)
#define LWE397_CTL_CTXSW_0_AUTO_ACK_AUTOACK                       (1)
#define LWE397_CTL_CTXSW_0_LWRR_CHANNEL                           15:12
#define LWE397_CTL_CTXSW_0_NEXT_CLASS                             25:16
#define LWE397_CTL_CTXSW_0_NEXT_CHANNEL                           31:28


// Register LWE397_CTL_STAT_0  
#define LWE397_CTL_STAT_0                                           (0xc)
#define LWE397_CTL_STAT_0_EN                                        0:0
#define LWE397_CTL_STAT_0_BLK                                       8:1
#define LWE397_CTL_STAT_0_BLK_IDX_HRD                               (0)
#define LWE397_CTL_STAT_0_BLK_IDX_HWR                               (1)
#define LWE397_CTL_STAT_0_BLK_IDX_MEMRDREQ                        (2)
#define LWE397_CTL_STAT_0_BLK_IDX_MEMRDRET                        (3)
#define LWE397_CTL_STAT_0_BLK_IDX_DMAACC                  (4)
#define LWE397_CTL_STAT_0_BLK_IDX_IDXFETCH                        (5)
#define LWE397_CTL_STAT_0_BLK_IDX_IDXLINES                        (6)
#define LWE397_CTL_STAT_0_BLK_IDX_IDXCOUNTS                       (7)
#define LWE397_CTL_STAT_0_BLK_IDX_VTXFETCH                        (8)
#define LWE397_CTL_STAT_0_BLK_IDX_VTXWORDS                        (9)
#define LWE397_CTL_STAT_0_BLK_IDX_VTXCACHE                        (10)
#define LWE397_CTL_STAT_0_BLK_IDX_VTXSTALL                        (11)
#define LWE397_CTL_STAT_0_BLK_IDX_PRIMSTALL                       (12)
#define LWE397_CTL_STAT_0_BLK_IDX_PRIMS                   (13)
#define LWE397_CTL_STAT_0_BLK_IDX_POINTS                  (14)
#define LWE397_CTL_STAT_0_BLK_IDX_LINES                   (15)
#define LWE397_CTL_STAT_0_BLK_IDX_TRIS                    (16)
#define LWE397_CTL_STAT_0_BLK_VPE_CLIPATADR                       (17)
#define LWE397_CTL_STAT_0_BLK_VPE_INSTISSUE                       (18)
#define LWE397_CTL_STAT_0_BLK_VPE_STALL                   (19)
#define LWE397_CTL_STAT_0_BLK_VPE_THREAD0                 (20)
#define LWE397_CTL_STAT_0_BLK_VPE_THREAD1                 (21)
#define LWE397_CTL_STAT_0_BLK_CLIP                        (22)
#define LWE397_CTL_STAT_0_BLK_CLIP_QRETIRE                        (23)
#define LWE397_CTL_STAT_0_BLK_CLIP_VPE                    (24)
#define LWE397_CTL_STAT_0_BLK_CLIP_ATRD                   (25)
#define LWE397_CTL_STAT_0_BLK_CLIP_TRIVREJ                        (26)
#define LWE397_CTL_STAT_0_BLK_CLIP_DEGEN                  (27)
#define LWE397_CTL_STAT_0_BLK_CLIP_ZEROAREA                       (28)
#define LWE397_CTL_STAT_0_BLK_CLIP_REJECT                 (29)
#define LWE397_CTL_STAT_0_BLK_CLIP_LWLL                   (30)
#define LWE397_CTL_STAT_0_BLK_CLIP_NEWTRI                 (31)
#define LWE397_CTL_STAT_0_BLK_CLIP_NEWLINE                        (32)
#define LWE397_CTL_STAT_0_BLK_CLIP_LR                     (33)
#define LWE397_CTL_STAT_0_BLK_CLIP_TB                     (34)
#define LWE397_CTL_STAT_0_BLK_CLIP_NF                     (35)
#define LWE397_CTL_STAT_0_BLK_CLIP_UC                     (36)
#define LWE397_CTL_STAT_0_BLK_SETUP_COV                   (37)
#define LWE397_CTL_STAT_0_BLK_SETUP_Z                     (38)
#define LWE397_CTL_STAT_0_BLK_QRAST_SRDAT0                        (39)
#define LWE397_CTL_STAT_0_BLK_QRAST_SRDAT1                        (40)
#define LWE397_CTL_STAT_0_BLK_QRAST_SB                    (41)
#define LWE397_CTL_STAT_0_BLK_QRAST_VRDAT0                        (42)
#define LWE397_CTL_STAT_0_BLK_QRAST_VRDAT1                        (43)
#define LWE397_CTL_STAT_0_BLK_QRAST_ZRDAT0                        (44)
#define LWE397_CTL_STAT_0_BLK_QRAST_ZRDAT1                        (45)
#define LWE397_CTL_STAT_0_BLK_QRAST_PSEQ                  (46)
#define LWE397_CTL_STAT_0_BLK_QRAST                       (47)
#define LWE397_CTL_STAT_0_BLK_QRAST_SBPIX                 (48)
#define LWE397_CTL_STAT_0_BLK_QRAST_QZLWLL                        (49)
#define LWE397_CTL_STAT_0_BLK_QRAST_QZKILL                        (50)
#define LWE397_CTL_STAT_0_BLK_QRAST_QSLWLL                        (51)
#define LWE397_CTL_STAT_0_BLK_QRAST_QVLWLL                        (52)
#define LWE397_CTL_STAT_0_BLK_QRAST_COARSE                        (53)
#define LWE397_CTL_STAT_0_BLK_QRAST_FINE                  (54)
#define LWE397_CTL_STAT_0_BLK_QRAST_CACHE0                        (55)
#define LWE397_CTL_STAT_0_BLK_QRAST_CACHE1                        (56)
#define LWE397_CTL_STAT_0_BLK_QRAST_PRIMS                 (57)
#define LWE397_CTL_STAT_0_BLK_QRAST_FLUSH                 (58)
#define LWE397_CTL_STAT_0_BLK_PSEQ_DWR                    (59)
#define LWE397_CTL_STAT_0_BLK_PSEQ_RDAT                   (60)
#define LWE397_CTL_STAT_0_BLK_PSEQ_RDATCMD                        (61)
#define LWE397_CTL_STAT_0_BLK_PSEQ                        (62)
#define LWE397_CTL_STAT_0_BLK_PSEQ_QPKT                   (63)
#define LWE397_CTL_STAT_0_BLK_PSEQ_QZS                    (64)
#define LWE397_CTL_STAT_0_BLK_PSEQ_PKT                    (65)
#define LWE397_CTL_STAT_0_BLK_PSEQ_ZS                     (66)
#define LWE397_CTL_STAT_0_BLK_PSEQ_NOP                    (67)
#define LWE397_CTL_STAT_0_BLK_PSEQ_CMD0                   (68)
#define LWE397_CTL_STAT_0_BLK_PSEQ_CMD1                   (69)
#define LWE397_CTL_STAT_0_BLK_PSEQ_LOAD                   (70)
#define LWE397_CTL_STAT_0_BLK_PSEQ_FETCH                  (71)
#define LWE397_CTL_STAT_0_BLK_PSEQ_FETCHNOEX                      (72)
#define LWE397_CTL_STAT_0_BLK_PSEQ_CACHE                  (73)
#define LWE397_CTL_STAT_0_BLK_PSEQ_RECIRC0                        (74)
#define LWE397_CTL_STAT_0_BLK_PSEQ_RECIRC1                        (75)
#define LWE397_CTL_STAT_0_BLK_PSEQ_RECIRC2                        (76)
#define LWE397_CTL_STAT_0_BLK_PSEQ_EPOCH                  (77)
#define LWE397_CTL_STAT_0_BLK_PSEQ_REQWAIT                        (78)
#define LWE397_CTL_STAT_0_BLK_PSEQ_RETWAIT                        (79)
#define LWE397_CTL_STAT_0_BLK_ATRAST_TRAM                 (80)
#define LWE397_CTL_STAT_0_BLK_ATRAST_SLOPE                        (81)
#define LWE397_CTL_STAT_0_BLK_ATRAST_TOP                  (82)
#define LWE397_CTL_STAT_0_BLK_ATRAST_BOT                  (83)
#define LWE397_CTL_STAT_0_BLK_TEX                 (84)
#define LWE397_CTL_STAT_0_BLK_TEX_MEMRDREQ                        (85)
#define LWE397_CTL_STAT_0_BLK_TEX_MEMRDRET                        (86)
#define LWE397_CTL_STAT_0_BLK_TEX_CACHEACC                        (87)
#define LWE397_CTL_STAT_0_BLK_TEX_CACHEMISS                       (88)
#define LWE397_CTL_STAT_0_BLK_TEX_FETCH                   (89)
#define LWE397_CTL_STAT_0_BLK_TEX_FETCHNOEX                       (90)
#define LWE397_CTL_STAT_0_BLK_TEX_FILTER0                 (91)
#define LWE397_CTL_STAT_0_BLK_TEX_FILTER1                 (92)
#define LWE397_CTL_STAT_0_BLK_TEX_LODHIST0                        (93)
#define LWE397_CTL_STAT_0_BLK_TEX_LODHIST1                        (94)
#define LWE397_CTL_STAT_0_BLK_TEX_ANISOHIST0                      (95)
#define LWE397_CTL_STAT_0_BLK_TEX_ANISOHIST1                      (96)
#define LWE397_CTL_STAT_0_BLK_TEX_ANISOFINE0                      (97)
#define LWE397_CTL_STAT_0_BLK_TEX_ANISOFINE1                      (98)
#define LWE397_CTL_STAT_0_BLK_TEX_LATHIDE                 (99)
#define LWE397_CTL_STAT_0_BLK_TEX_PERFDEG                 (100)
#define LWE397_CTL_STAT_0_BLK_ALU_TOP                     (101)
#define LWE397_CTL_STAT_0_BLK_ALU_BOT                     (102)
#define LWE397_CTL_STAT_0_BLK_DWR                 (103)
#define LWE397_CTL_STAT_0_BLK_DWR_STORE                   (104)
#define LWE397_CTL_STAT_0_BLK_DWR_WRITE                   (105)
#define LWE397_CTL_STAT_0_BLK_DWR_WRITENOEX                       (106)
#define LWE397_CTL_STAT_0_BLK_DWR_SPILL                   (107)
#define LWE397_CTL_STAT_0_BLK_DWR_FLUSH                   (108)
#define LWE397_CTL_STAT_0_BLK_DWR_LATEOPS                 (109)
#define LWE397_CTL_STAT_0_BLK_DWR_CACHE                   (110)
#define LWE397_CTL_STAT_0_BLK_FDC_DWR0                    (111)
#define LWE397_CTL_STAT_0_BLK_FDC_DWR1                    (112)
#define LWE397_CTL_STAT_0_BLK_FDC_IMEMRDREQ                       (113)
#define LWE397_CTL_STAT_0_BLK_FDC_EMEMRDREQ                       (114)
#define LWE397_CTL_STAT_0_BLK_FDC_IMEMWRREQ                       (115)
#define LWE397_CTL_STAT_0_BLK_FDC_EMEMWRREQ                       (116)
#define LWE397_CTL_STAT_0_BLK_FDC_PSEQ0                   (117)
#define LWE397_CTL_STAT_0_BLK_FDC_PSEQ1                   (118)
#define LWE397_CTL_STAT_0_BLK_FDC_SBPSEQ                  (119)
#define LWE397_CTL_STAT_0_BLK_FDC_QSRD0                   (120)
#define LWE397_CTL_STAT_0_BLK_FDC_QSRD1                   (121)
#define LWE397_CTL_STAT_0_BLK_FDC_QSWR0                   (122)
#define LWE397_CTL_STAT_0_BLK_FDC_QSWR1                   (123)
#define LWE397_CTL_STAT_0_BLK_FDC_SBQRAST                 (124)
#define LWE397_CTL_STAT_0_BLK_FDC_QVRD0                   (125)
#define LWE397_CTL_STAT_0_BLK_FDC_QVRD1                   (126)
#define LWE397_CTL_STAT_0_BLK_FDC_QVWR0                   (127)
#define LWE397_CTL_STAT_0_BLK_FDC_QVWR1                   (128)
#define LWE397_CTL_STAT_0_BLK_FDC_QZRD0                   (129)
#define LWE397_CTL_STAT_0_BLK_FDC_QZRD1                   (130)
#define LWE397_CTL_STAT_0_BLK_FDC_QZWR0                   (131)
#define LWE397_CTL_STAT_0_BLK_FDC_QZWR1                   (132)
#define LWE397_CTL_STAT_0_BLK_FDC_L2RD0                   (133)
#define LWE397_CTL_STAT_0_BLK_FDC_L2RD1                   (134)
#define LWE397_CTL_STAT_0_BLK_FDC_L2RD2                   (135)
#define LWE397_CTL_STAT_0_BLK_FDC_L2RD3                   (136)
#define LWE397_CTL_STAT_0_BLK_FDC_L2RD4                   (137)
#define LWE397_CTL_STAT_0_BLK_FDC_L2RD5                   (138)
#define LWE397_CTL_STAT_0_BLK_FDC_L2RD6                   (139)
#define LWE397_CTL_STAT_0_BLK_FDC_L2RD7                   (140)
#define LWE397_CTL_STAT_0_BLK_FDC_L2WR0                   (141)
#define LWE397_CTL_STAT_0_BLK_FDC_L2WR1                   (142)
#define LWE397_CTL_STAT_0_BLK_FDC_L2WR2                   (143)
#define LWE397_CTL_STAT_0_BLK_FDC_L2WR3                   (144)
#define LWE397_CTL_STAT_0_BLK_FDC_L2WR4                   (145)
#define LWE397_CTL_STAT_0_BLK_FDC_L2WR5                   (146)
#define LWE397_CTL_STAT_0_BLK_FDC_L2WR6                   (147)
#define LWE397_CTL_STAT_0_BLK_FDC_L2WR7                   (148)
#define LWE397_CTL_STAT_0_BLK_FDC_L2MC                    (149)
#define LWE397_CTL_STAT_0_BLK_FDC_L2RDEVICT0                      (150)
#define LWE397_CTL_STAT_0_BLK_FDC_L2RDEVICT1                      (151)
#define LWE397_CTL_STAT_0_BLK_FDC_L2WREVICT0                      (152)
#define LWE397_CTL_STAT_0_BLK_FDC_L2WREVICT1                      (153)
#define LWE397_CTL_STAT_0_BLK_FDC_WRSNOOP                 (154)
#define LWE397_CTL_STAT_0_BLK_FDC_SBSNOOP                 (155)
#define LWE397_CTL_STAT_0_BLK_FDC_RDTAGSTALL0                     (156)
#define LWE397_CTL_STAT_0_BLK_FDC_RDTAGSTALL1                     (157)
#define LWE397_CTL_STAT_0_BLK_FDC_WRTAGSTALL0                     (158)
#define LWE397_CTL_STAT_0_BLK_FDC_WRTAGSTALL1                     (159)
#define LWE397_CTL_STAT_0_BLK_NUM_BLOCKS                  (160)

#define LWE397_CTL_STAT_0_OVF                       31:31


// Register LWE397_CTL_STAT  
#define LWE397_CTL_STAT                   (0xc)
#define LWE397_CTL_STAT_EN                  0:0

#define LWE397_CTL_STAT_BLK                 8:1
#define LWE397_CTL_STAT_BLK_IDX_HRD                       (0)
#define LWE397_CTL_STAT_BLK_IDX_HWR                       (1)
#define LWE397_CTL_STAT_BLK_IDX_MEMRDREQ                  (2)
#define LWE397_CTL_STAT_BLK_IDX_MEMRDRET                  (3)
#define LWE397_CTL_STAT_BLK_IDX_DMAACC                    (4)
#define LWE397_CTL_STAT_BLK_IDX_IDXFETCH                  (5)
#define LWE397_CTL_STAT_BLK_IDX_IDXLINES                  (6)
#define LWE397_CTL_STAT_BLK_IDX_IDXCOUNTS                 (7)
#define LWE397_CTL_STAT_BLK_IDX_VTXFETCH                  (8)
#define LWE397_CTL_STAT_BLK_IDX_VTXWORDS                  (9)
#define LWE397_CTL_STAT_BLK_IDX_VTXCACHE                  (10)
#define LWE397_CTL_STAT_BLK_IDX_VTXSTALL                  (11)
#define LWE397_CTL_STAT_BLK_IDX_PRIMSTALL                 (12)
#define LWE397_CTL_STAT_BLK_IDX_PRIMS                     (13)
#define LWE397_CTL_STAT_BLK_IDX_POINTS                    (14)
#define LWE397_CTL_STAT_BLK_IDX_LINES                     (15)
#define LWE397_CTL_STAT_BLK_IDX_TRIS                      (16)
#define LWE397_CTL_STAT_BLK_VPE_CLIPATADR                 (17)
#define LWE397_CTL_STAT_BLK_VPE_INSTISSUE                 (18)
#define LWE397_CTL_STAT_BLK_VPE_STALL                     (19)
#define LWE397_CTL_STAT_BLK_VPE_THREAD0                   (20)
#define LWE397_CTL_STAT_BLK_VPE_THREAD1                   (21)
#define LWE397_CTL_STAT_BLK_CLIP                          (22)
#define LWE397_CTL_STAT_BLK_CLIP_QRETIRE                  (23)
#define LWE397_CTL_STAT_BLK_CLIP_VPE                      (24)
#define LWE397_CTL_STAT_BLK_CLIP_ATRD                     (25)
#define LWE397_CTL_STAT_BLK_CLIP_TRIVREJ                  (26)
#define LWE397_CTL_STAT_BLK_CLIP_DEGEN                    (27)
#define LWE397_CTL_STAT_BLK_CLIP_ZEROAREA                 (28)
#define LWE397_CTL_STAT_BLK_CLIP_REJECT                   (29)
#define LWE397_CTL_STAT_BLK_CLIP_LWLL                     (30)
#define LWE397_CTL_STAT_BLK_CLIP_NEWTRI                   (31)
#define LWE397_CTL_STAT_BLK_CLIP_NEWLINE                  (32)
#define LWE397_CTL_STAT_BLK_CLIP_LR                       (33)
#define LWE397_CTL_STAT_BLK_CLIP_TB                       (34)
#define LWE397_CTL_STAT_BLK_CLIP_NF                       (35)
#define LWE397_CTL_STAT_BLK_CLIP_UC                       (36)
#define LWE397_CTL_STAT_BLK_SETUP_COV                     (37)
#define LWE397_CTL_STAT_BLK_SETUP_Z                       (38)
#define LWE397_CTL_STAT_BLK_QRAST_SRDAT0                  (39)
#define LWE397_CTL_STAT_BLK_QRAST_SRDAT1                  (40)
#define LWE397_CTL_STAT_BLK_QRAST_SB                      (41)
#define LWE397_CTL_STAT_BLK_QRAST_VRDAT0                  (42)
#define LWE397_CTL_STAT_BLK_QRAST_VRDAT1                  (43)
#define LWE397_CTL_STAT_BLK_QRAST_ZRDAT0                  (44)
#define LWE397_CTL_STAT_BLK_QRAST_ZRDAT1                  (45)
#define LWE397_CTL_STAT_BLK_QRAST_PSEQ                    (46)
#define LWE397_CTL_STAT_BLK_QRAST                         (47)
#define LWE397_CTL_STAT_BLK_QRAST_SBPIX                   (48)
#define LWE397_CTL_STAT_BLK_QRAST_QZLWLL                  (49)
#define LWE397_CTL_STAT_BLK_QRAST_QZKILL                  (50)
#define LWE397_CTL_STAT_BLK_QRAST_QSLWLL                  (51)
#define LWE397_CTL_STAT_BLK_QRAST_QVLWLL                  (52)
#define LWE397_CTL_STAT_BLK_QRAST_COARSE                  (53)
#define LWE397_CTL_STAT_BLK_QRAST_FINE                    (54)
#define LWE397_CTL_STAT_BLK_QRAST_CACHE0                  (55)
#define LWE397_CTL_STAT_BLK_QRAST_CACHE1                  (56)
#define LWE397_CTL_STAT_BLK_QRAST_PRIMS                   (57)
#define LWE397_CTL_STAT_BLK_QRAST_FLUSH                   (58)
#define LWE397_CTL_STAT_BLK_PSEQ_DWR                      (59)
#define LWE397_CTL_STAT_BLK_PSEQ_RDAT                     (60)
#define LWE397_CTL_STAT_BLK_PSEQ_RDATCMD                  (61)
#define LWE397_CTL_STAT_BLK_PSEQ                          (62)
#define LWE397_CTL_STAT_BLK_PSEQ_QPKT                     (63)
#define LWE397_CTL_STAT_BLK_PSEQ_QZS                      (64)
#define LWE397_CTL_STAT_BLK_PSEQ_PKT                      (65)
#define LWE397_CTL_STAT_BLK_PSEQ_ZS                       (66)
#define LWE397_CTL_STAT_BLK_PSEQ_NOP                      (67)
#define LWE397_CTL_STAT_BLK_PSEQ_CMD0                     (68)
#define LWE397_CTL_STAT_BLK_PSEQ_CMD1                     (69)
#define LWE397_CTL_STAT_BLK_PSEQ_LOAD                     (70)
#define LWE397_CTL_STAT_BLK_PSEQ_FETCH                    (71)
#define LWE397_CTL_STAT_BLK_PSEQ_FETCHNOEX                (72)
#define LWE397_CTL_STAT_BLK_PSEQ_CACHE                    (73)
#define LWE397_CTL_STAT_BLK_PSEQ_RECIRC0                  (74)
#define LWE397_CTL_STAT_BLK_PSEQ_RECIRC1                  (75)
#define LWE397_CTL_STAT_BLK_PSEQ_RECIRC2                  (76)
#define LWE397_CTL_STAT_BLK_PSEQ_EPOCH                    (77)
#define LWE397_CTL_STAT_BLK_PSEQ_REQWAIT                  (78)
#define LWE397_CTL_STAT_BLK_PSEQ_RETWAIT                  (79)
#define LWE397_CTL_STAT_BLK_ATRAST_TRAM                   (80)
#define LWE397_CTL_STAT_BLK_ATRAST_SLOPE                  (81)
#define LWE397_CTL_STAT_BLK_ATRAST_TOP                    (82)
#define LWE397_CTL_STAT_BLK_ATRAST_BOT                    (83)
#define LWE397_CTL_STAT_BLK_TEX                           (84)
#define LWE397_CTL_STAT_BLK_TEX_MEMRDREQ                  (85)
#define LWE397_CTL_STAT_BLK_TEX_MEMRDRET                  (86)
#define LWE397_CTL_STAT_BLK_TEX_CACHEACC                  (87)
#define LWE397_CTL_STAT_BLK_TEX_CACHEMISS                 (88)
#define LWE397_CTL_STAT_BLK_TEX_FETCH                     (89)
#define LWE397_CTL_STAT_BLK_TEX_FETCHNOEX                 (90)
#define LWE397_CTL_STAT_BLK_TEX_FILTER0                   (91)
#define LWE397_CTL_STAT_BLK_TEX_FILTER1                   (92)
#define LWE397_CTL_STAT_BLK_TEX_LODHIST0                  (93)
#define LWE397_CTL_STAT_BLK_TEX_LODHIST1                  (94)
#define LWE397_CTL_STAT_BLK_TEX_ANISOHIST0                (95)
#define LWE397_CTL_STAT_BLK_TEX_ANISOHIST1                (96)
#define LWE397_CTL_STAT_BLK_TEX_ANISOFINE0                (97)
#define LWE397_CTL_STAT_BLK_TEX_ANISOFINE1                (98)
#define LWE397_CTL_STAT_BLK_TEX_LATHIDE                   (99)
#define LWE397_CTL_STAT_BLK_TEX_PERFDEG                   (100)
#define LWE397_CTL_STAT_BLK_ALU_TOP                       (101)
#define LWE397_CTL_STAT_BLK_ALU_BOT                       (102)
#define LWE397_CTL_STAT_BLK_DWR                           (103)
#define LWE397_CTL_STAT_BLK_DWR_STORE                     (104)
#define LWE397_CTL_STAT_BLK_DWR_WRITE                     (105)
#define LWE397_CTL_STAT_BLK_DWR_WRITENOEX                 (106)
#define LWE397_CTL_STAT_BLK_DWR_SPILL                     (107)
#define LWE397_CTL_STAT_BLK_DWR_FLUSH                     (108)
#define LWE397_CTL_STAT_BLK_DWR_LATEOPS                   (109)
#define LWE397_CTL_STAT_BLK_DWR_CACHE                     (110)
#define LWE397_CTL_STAT_BLK_FDC_DWR0                      (111)
#define LWE397_CTL_STAT_BLK_FDC_DWR1                      (112)
#define LWE397_CTL_STAT_BLK_FDC_IMEMRDREQ                 (113)
#define LWE397_CTL_STAT_BLK_FDC_EMEMRDREQ                 (114)
#define LWE397_CTL_STAT_BLK_FDC_IMEMWRREQ                 (115)
#define LWE397_CTL_STAT_BLK_FDC_EMEMWRREQ                 (116)
#define LWE397_CTL_STAT_BLK_FDC_PSEQ0                     (117)
#define LWE397_CTL_STAT_BLK_FDC_PSEQ1                     (118)
#define LWE397_CTL_STAT_BLK_FDC_SBPSEQ                    (119)
#define LWE397_CTL_STAT_BLK_FDC_QSRD0                     (120)
#define LWE397_CTL_STAT_BLK_FDC_QSRD1                     (121)
#define LWE397_CTL_STAT_BLK_FDC_QSWR0                     (122)
#define LWE397_CTL_STAT_BLK_FDC_QSWR1                     (123)
#define LWE397_CTL_STAT_BLK_FDC_SBQRAST                   (124)
#define LWE397_CTL_STAT_BLK_FDC_QVRD0                     (125)
#define LWE397_CTL_STAT_BLK_FDC_QVRD1                     (126)
#define LWE397_CTL_STAT_BLK_FDC_QVWR0                     (127)
#define LWE397_CTL_STAT_BLK_FDC_QVWR1                     (128)
#define LWE397_CTL_STAT_BLK_FDC_QZRD0                     (129)
#define LWE397_CTL_STAT_BLK_FDC_QZRD1                     (130)
#define LWE397_CTL_STAT_BLK_FDC_QZWR0                     (131)
#define LWE397_CTL_STAT_BLK_FDC_QZWR1                     (132)
#define LWE397_CTL_STAT_BLK_FDC_L2RD0                     (133)
#define LWE397_CTL_STAT_BLK_FDC_L2RD1                     (134)
#define LWE397_CTL_STAT_BLK_FDC_L2RD2                     (135)
#define LWE397_CTL_STAT_BLK_FDC_L2RD3                     (136)
#define LWE397_CTL_STAT_BLK_FDC_L2RD4                     (137)
#define LWE397_CTL_STAT_BLK_FDC_L2RD5                     (138)
#define LWE397_CTL_STAT_BLK_FDC_L2RD6                     (139)
#define LWE397_CTL_STAT_BLK_FDC_L2RD7                     (140)
#define LWE397_CTL_STAT_BLK_FDC_L2WR0                     (141)
#define LWE397_CTL_STAT_BLK_FDC_L2WR1                     (142)
#define LWE397_CTL_STAT_BLK_FDC_L2WR2                     (143)
#define LWE397_CTL_STAT_BLK_FDC_L2WR3                     (144)
#define LWE397_CTL_STAT_BLK_FDC_L2WR4                     (145)
#define LWE397_CTL_STAT_BLK_FDC_L2WR5                     (146)
#define LWE397_CTL_STAT_BLK_FDC_L2WR6                     (147)
#define LWE397_CTL_STAT_BLK_FDC_L2WR7                     (148)
#define LWE397_CTL_STAT_BLK_FDC_L2MC                      (149)
#define LWE397_CTL_STAT_BLK_FDC_L2RDEVICT0                        (150)
#define LWE397_CTL_STAT_BLK_FDC_L2RDEVICT1                        (151)
#define LWE397_CTL_STAT_BLK_FDC_L2WREVICT0                        (152)
#define LWE397_CTL_STAT_BLK_FDC_L2WREVICT1                        (153)
#define LWE397_CTL_STAT_BLK_FDC_WRSNOOP                   (154)
#define LWE397_CTL_STAT_BLK_FDC_SBSNOOP                   (155)
#define LWE397_CTL_STAT_BLK_FDC_RDTAGSTALL0                       (156)
#define LWE397_CTL_STAT_BLK_FDC_RDTAGSTALL1                       (157)
#define LWE397_CTL_STAT_BLK_FDC_WRTAGSTALL0                       (158)
#define LWE397_CTL_STAT_BLK_FDC_WRTAGSTALL1                       (159)
#define LWE397_CTL_STAT_BLK_NUM_BLOCKS                    (160)

#define LWE397_CTL_STAT_OVF                 31:31


// Register LWE397_CTL_STAT_1  
#define LWE397_CTL_STAT_1                 (0xd)
#define LWE397_CTL_STAT_1_EN                        0:0

#define LWE397_CTL_STAT_1_BLK                       8:1
#define LWE397_CTL_STAT_1_BLK_IDX_HRD                     (0)
#define LWE397_CTL_STAT_1_BLK_IDX_HWR                     (1)
#define LWE397_CTL_STAT_1_BLK_IDX_MEMRDREQ                        (2)
#define LWE397_CTL_STAT_1_BLK_IDX_MEMRDRET                        (3)
#define LWE397_CTL_STAT_1_BLK_IDX_DMAACC                  (4)
#define LWE397_CTL_STAT_1_BLK_IDX_IDXFETCH                        (5)
#define LWE397_CTL_STAT_1_BLK_IDX_IDXLINES                        (6)
#define LWE397_CTL_STAT_1_BLK_IDX_IDXCOUNTS                       (7)
#define LWE397_CTL_STAT_1_BLK_IDX_VTXFETCH                        (8)
#define LWE397_CTL_STAT_1_BLK_IDX_VTXWORDS                        (9)
#define LWE397_CTL_STAT_1_BLK_IDX_VTXCACHE                        (10)
#define LWE397_CTL_STAT_1_BLK_IDX_VTXSTALL                        (11)
#define LWE397_CTL_STAT_1_BLK_IDX_PRIMSTALL                       (12)
#define LWE397_CTL_STAT_1_BLK_IDX_PRIMS                   (13)
#define LWE397_CTL_STAT_1_BLK_IDX_POINTS                  (14)
#define LWE397_CTL_STAT_1_BLK_IDX_LINES                   (15)
#define LWE397_CTL_STAT_1_BLK_IDX_TRIS                    (16)
#define LWE397_CTL_STAT_1_BLK_VPE_CLIPATADR                       (17)
#define LWE397_CTL_STAT_1_BLK_VPE_INSTISSUE                       (18)
#define LWE397_CTL_STAT_1_BLK_VPE_STALL                   (19)
#define LWE397_CTL_STAT_1_BLK_VPE_THREAD0                 (20)
#define LWE397_CTL_STAT_1_BLK_VPE_THREAD1                 (21)
#define LWE397_CTL_STAT_1_BLK_CLIP                        (22)
#define LWE397_CTL_STAT_1_BLK_CLIP_QRETIRE                        (23)
#define LWE397_CTL_STAT_1_BLK_CLIP_VPE                    (24)
#define LWE397_CTL_STAT_1_BLK_CLIP_ATRD                   (25)
#define LWE397_CTL_STAT_1_BLK_CLIP_TRIVREJ                        (26)
#define LWE397_CTL_STAT_1_BLK_CLIP_DEGEN                  (27)
#define LWE397_CTL_STAT_1_BLK_CLIP_ZEROAREA                       (28)
#define LWE397_CTL_STAT_1_BLK_CLIP_REJECT                 (29)
#define LWE397_CTL_STAT_1_BLK_CLIP_LWLL                   (30)
#define LWE397_CTL_STAT_1_BLK_CLIP_NEWTRI                 (31)
#define LWE397_CTL_STAT_1_BLK_CLIP_NEWLINE                        (32)
#define LWE397_CTL_STAT_1_BLK_CLIP_LR                     (33)
#define LWE397_CTL_STAT_1_BLK_CLIP_TB                     (34)
#define LWE397_CTL_STAT_1_BLK_CLIP_NF                     (35)
#define LWE397_CTL_STAT_1_BLK_CLIP_UC                     (36)
#define LWE397_CTL_STAT_1_BLK_SETUP_COV                   (37)
#define LWE397_CTL_STAT_1_BLK_SETUP_Z                     (38)
#define LWE397_CTL_STAT_1_BLK_QRAST_SRDAT0                        (39)
#define LWE397_CTL_STAT_1_BLK_QRAST_SRDAT1                        (40)
#define LWE397_CTL_STAT_1_BLK_QRAST_SB                    (41)
#define LWE397_CTL_STAT_1_BLK_QRAST_VRDAT0                        (42)
#define LWE397_CTL_STAT_1_BLK_QRAST_VRDAT1                        (43)
#define LWE397_CTL_STAT_1_BLK_QRAST_ZRDAT0                        (44)
#define LWE397_CTL_STAT_1_BLK_QRAST_ZRDAT1                        (45)
#define LWE397_CTL_STAT_1_BLK_QRAST_PSEQ                  (46)
#define LWE397_CTL_STAT_1_BLK_QRAST                       (47)
#define LWE397_CTL_STAT_1_BLK_QRAST_SBPIX                 (48)
#define LWE397_CTL_STAT_1_BLK_QRAST_QZLWLL                        (49)
#define LWE397_CTL_STAT_1_BLK_QRAST_QZKILL                        (50)
#define LWE397_CTL_STAT_1_BLK_QRAST_QSLWLL                        (51)
#define LWE397_CTL_STAT_1_BLK_QRAST_QVLWLL                        (52)
#define LWE397_CTL_STAT_1_BLK_QRAST_COARSE                        (53)
#define LWE397_CTL_STAT_1_BLK_QRAST_FINE                  (54)
#define LWE397_CTL_STAT_1_BLK_QRAST_CACHE0                        (55)
#define LWE397_CTL_STAT_1_BLK_QRAST_CACHE1                        (56)
#define LWE397_CTL_STAT_1_BLK_QRAST_PRIMS                 (57)
#define LWE397_CTL_STAT_1_BLK_QRAST_FLUSH                 (58)
#define LWE397_CTL_STAT_1_BLK_PSEQ_DWR                    (59)
#define LWE397_CTL_STAT_1_BLK_PSEQ_RDAT                   (60)
#define LWE397_CTL_STAT_1_BLK_PSEQ_RDATCMD                        (61)
#define LWE397_CTL_STAT_1_BLK_PSEQ                        (62)
#define LWE397_CTL_STAT_1_BLK_PSEQ_QPKT                   (63)
#define LWE397_CTL_STAT_1_BLK_PSEQ_QZS                    (64)
#define LWE397_CTL_STAT_1_BLK_PSEQ_PKT                    (65)
#define LWE397_CTL_STAT_1_BLK_PSEQ_ZS                     (66)
#define LWE397_CTL_STAT_1_BLK_PSEQ_NOP                    (67)
#define LWE397_CTL_STAT_1_BLK_PSEQ_CMD0                   (68)
#define LWE397_CTL_STAT_1_BLK_PSEQ_CMD1                   (69)
#define LWE397_CTL_STAT_1_BLK_PSEQ_LOAD                   (70)
#define LWE397_CTL_STAT_1_BLK_PSEQ_FETCH                  (71)
#define LWE397_CTL_STAT_1_BLK_PSEQ_FETCHNOEX                      (72)
#define LWE397_CTL_STAT_1_BLK_PSEQ_CACHE                  (73)
#define LWE397_CTL_STAT_1_BLK_PSEQ_RECIRC0                        (74)
#define LWE397_CTL_STAT_1_BLK_PSEQ_RECIRC1                        (75)
#define LWE397_CTL_STAT_1_BLK_PSEQ_RECIRC2                        (76)
#define LWE397_CTL_STAT_1_BLK_PSEQ_EPOCH                  (77)
#define LWE397_CTL_STAT_1_BLK_PSEQ_REQWAIT                        (78)
#define LWE397_CTL_STAT_1_BLK_PSEQ_RETWAIT                        (79)
#define LWE397_CTL_STAT_1_BLK_ATRAST_TRAM                 (80)
#define LWE397_CTL_STAT_1_BLK_ATRAST_SLOPE                        (81)
#define LWE397_CTL_STAT_1_BLK_ATRAST_TOP                  (82)
#define LWE397_CTL_STAT_1_BLK_ATRAST_BOT                  (83)
#define LWE397_CTL_STAT_1_BLK_TEX                 (84)
#define LWE397_CTL_STAT_1_BLK_TEX_MEMRDREQ                        (85)
#define LWE397_CTL_STAT_1_BLK_TEX_MEMRDRET                        (86)
#define LWE397_CTL_STAT_1_BLK_TEX_CACHEACC                        (87)
#define LWE397_CTL_STAT_1_BLK_TEX_CACHEMISS                       (88)
#define LWE397_CTL_STAT_1_BLK_TEX_FETCH                   (89)
#define LWE397_CTL_STAT_1_BLK_TEX_FETCHNOEX                       (90)
#define LWE397_CTL_STAT_1_BLK_TEX_FILTER0                 (91)
#define LWE397_CTL_STAT_1_BLK_TEX_FILTER1                 (92)
#define LWE397_CTL_STAT_1_BLK_TEX_LODHIST0                        (93)
#define LWE397_CTL_STAT_1_BLK_TEX_LODHIST1                        (94)
#define LWE397_CTL_STAT_1_BLK_TEX_ANISOHIST0                      (95)
#define LWE397_CTL_STAT_1_BLK_TEX_ANISOHIST1                      (96)
#define LWE397_CTL_STAT_1_BLK_TEX_ANISOFINE0                      (97)
#define LWE397_CTL_STAT_1_BLK_TEX_ANISOFINE1                      (98)
#define LWE397_CTL_STAT_1_BLK_TEX_LATHIDE                 (99)
#define LWE397_CTL_STAT_1_BLK_TEX_PERFDEG                 (100)
#define LWE397_CTL_STAT_1_BLK_ALU_TOP                     (101)
#define LWE397_CTL_STAT_1_BLK_ALU_BOT                     (102)
#define LWE397_CTL_STAT_1_BLK_DWR                 (103)
#define LWE397_CTL_STAT_1_BLK_DWR_STORE                   (104)
#define LWE397_CTL_STAT_1_BLK_DWR_WRITE                   (105)
#define LWE397_CTL_STAT_1_BLK_DWR_WRITENOEX                       (106)
#define LWE397_CTL_STAT_1_BLK_DWR_SPILL                   (107)
#define LWE397_CTL_STAT_1_BLK_DWR_FLUSH                   (108)
#define LWE397_CTL_STAT_1_BLK_DWR_LATEOPS                 (109)
#define LWE397_CTL_STAT_1_BLK_DWR_CACHE                   (110)
#define LWE397_CTL_STAT_1_BLK_FDC_DWR0                    (111)
#define LWE397_CTL_STAT_1_BLK_FDC_DWR1                    (112)
#define LWE397_CTL_STAT_1_BLK_FDC_IMEMRDREQ                       (113)
#define LWE397_CTL_STAT_1_BLK_FDC_EMEMRDREQ                       (114)
#define LWE397_CTL_STAT_1_BLK_FDC_IMEMWRREQ                       (115)
#define LWE397_CTL_STAT_1_BLK_FDC_EMEMWRREQ                       (116)
#define LWE397_CTL_STAT_1_BLK_FDC_PSEQ0                   (117)
#define LWE397_CTL_STAT_1_BLK_FDC_PSEQ1                   (118)
#define LWE397_CTL_STAT_1_BLK_FDC_SBPSEQ                  (119)
#define LWE397_CTL_STAT_1_BLK_FDC_QSRD0                   (120)
#define LWE397_CTL_STAT_1_BLK_FDC_QSRD1                   (121)
#define LWE397_CTL_STAT_1_BLK_FDC_QSWR0                   (122)
#define LWE397_CTL_STAT_1_BLK_FDC_QSWR1                   (123)
#define LWE397_CTL_STAT_1_BLK_FDC_SBQRAST                 (124)
#define LWE397_CTL_STAT_1_BLK_FDC_QVRD0                   (125)
#define LWE397_CTL_STAT_1_BLK_FDC_QVRD1                   (126)
#define LWE397_CTL_STAT_1_BLK_FDC_QVWR0                   (127)
#define LWE397_CTL_STAT_1_BLK_FDC_QVWR1                   (128)
#define LWE397_CTL_STAT_1_BLK_FDC_QZRD0                   (129)
#define LWE397_CTL_STAT_1_BLK_FDC_QZRD1                   (130)
#define LWE397_CTL_STAT_1_BLK_FDC_QZWR0                   (131)
#define LWE397_CTL_STAT_1_BLK_FDC_QZWR1                   (132)
#define LWE397_CTL_STAT_1_BLK_FDC_L2RD0                   (133)
#define LWE397_CTL_STAT_1_BLK_FDC_L2RD1                   (134)
#define LWE397_CTL_STAT_1_BLK_FDC_L2RD2                   (135)
#define LWE397_CTL_STAT_1_BLK_FDC_L2RD3                   (136)
#define LWE397_CTL_STAT_1_BLK_FDC_L2RD4                   (137)
#define LWE397_CTL_STAT_1_BLK_FDC_L2RD5                   (138)
#define LWE397_CTL_STAT_1_BLK_FDC_L2RD6                   (139)
#define LWE397_CTL_STAT_1_BLK_FDC_L2RD7                   (140)
#define LWE397_CTL_STAT_1_BLK_FDC_L2WR0                   (141)
#define LWE397_CTL_STAT_1_BLK_FDC_L2WR1                   (142)
#define LWE397_CTL_STAT_1_BLK_FDC_L2WR2                   (143)
#define LWE397_CTL_STAT_1_BLK_FDC_L2WR3                   (144)
#define LWE397_CTL_STAT_1_BLK_FDC_L2WR4                   (145)
#define LWE397_CTL_STAT_1_BLK_FDC_L2WR5                   (146)
#define LWE397_CTL_STAT_1_BLK_FDC_L2WR6                   (147)
#define LWE397_CTL_STAT_1_BLK_FDC_L2WR7                   (148)
#define LWE397_CTL_STAT_1_BLK_FDC_L2MC                    (149)
#define LWE397_CTL_STAT_1_BLK_FDC_L2RDEVICT0                      (150)
#define LWE397_CTL_STAT_1_BLK_FDC_L2RDEVICT1                      (151)
#define LWE397_CTL_STAT_1_BLK_FDC_L2WREVICT0                      (152)
#define LWE397_CTL_STAT_1_BLK_FDC_L2WREVICT1                      (153)
#define LWE397_CTL_STAT_1_BLK_FDC_WRSNOOP                 (154)
#define LWE397_CTL_STAT_1_BLK_FDC_SBSNOOP                 (155)
#define LWE397_CTL_STAT_1_BLK_FDC_RDTAGSTALL0                     (156)
#define LWE397_CTL_STAT_1_BLK_FDC_RDTAGSTALL1                     (157)
#define LWE397_CTL_STAT_1_BLK_FDC_WRTAGSTALL0                     (158)
#define LWE397_CTL_STAT_1_BLK_FDC_WRTAGSTALL1                     (159)
#define LWE397_CTL_STAT_1_BLK_NUM_BLOCKS                  (160)

#define LWE397_CTL_STAT_1_OVF                       31:31


// Register LWE397_CTL_STAT_CLK_COUNT_0  
#define LWE397_CTL_STAT_CLK_COUNT_0                       (0xe)
#define LWE397_CTL_STAT_CLK_COUNT_0_VAL                     31:0


// Register LWE397_CTL_STAT_CLK_COUNT  
#define LWE397_CTL_STAT_CLK_COUNT                 (0xe)
#define LWE397_CTL_STAT_CLK_COUNT_VAL                       31:0


// Register LWE397_CTL_STAT_CLK_COUNT_1  
#define LWE397_CTL_STAT_CLK_COUNT_1                       (0xf)
#define LWE397_CTL_STAT_CLK_COUNT_1_VAL                     31:0


// Register LWE397_CTL_STAT_XFER_COUNT_0  
#define LWE397_CTL_STAT_XFER_COUNT_0                      (0x10)
#define LWE397_CTL_STAT_XFER_COUNT_0_VAL                    31:0


// Register LWE397_CTL_STAT_XFER_COUNT  
#define LWE397_CTL_STAT_XFER_COUNT                        (0x10)
#define LWE397_CTL_STAT_XFER_COUNT_VAL                      31:0


// Register LWE397_CTL_STAT_XFER_COUNT_1  
#define LWE397_CTL_STAT_XFER_COUNT_1                      (0x11)
#define LWE397_CTL_STAT_XFER_COUNT_1_VAL                    31:0


// Register LWE397_CTL_STAT_WAIT_COUNT_0  
#define LWE397_CTL_STAT_WAIT_COUNT_0                      (0x12)
#define LWE397_CTL_STAT_WAIT_COUNT_0_VAL                    31:0


// Register LWE397_CTL_STAT_WAIT_COUNT  
#define LWE397_CTL_STAT_WAIT_COUNT                        (0x12)
#define LWE397_CTL_STAT_WAIT_COUNT_VAL                      31:0


// Register LWE397_CTL_STAT_WAIT_COUNT_1  
#define LWE397_CTL_STAT_WAIT_COUNT_1                      (0x13)
#define LWE397_CTL_STAT_WAIT_COUNT_1_VAL                    31:0


// Register LWE397_CTL_STAT_EN_COUNT_0  
#define LWE397_CTL_STAT_EN_COUNT_0                        (0x14)
#define LWE397_CTL_STAT_EN_COUNT_0_VAL                      31:0


// Register LWE397_CTL_STAT_EN_COUNT  
#define LWE397_CTL_STAT_EN_COUNT                  (0x14)
#define LWE397_CTL_STAT_EN_COUNT_VAL                        31:0


// Register LWE397_CTL_STAT_EN_COUNT_1  
#define LWE397_CTL_STAT_EN_COUNT_1                        (0x15)
#define LWE397_CTL_STAT_EN_COUNT_1_VAL                      31:0

#define NUM_ATTR        16

// Register LWE397_IDX_ATTRIBUTE_0  
#define LWE397_IDX_ATTRIBUTE_0                    (0x100)
#define LWE397_IDX_ATTRIBUTE_0_ATTR_BASE                    31:0

#define LWE397_IDX_ATTRIBUTE_0_SNX_ZERO_PRESERVE                    20:20
#define LWE397_IDX_ATTRIBUTE_0_SNX_ZERO_PRESERVE_DISABLE                  (0)
#define LWE397_IDX_ATTRIBUTE_0_SNX_ZERO_PRESERVE_ENABLE                   (1)

#define LWE397_IDX_ATTRIBUTE_0_ATTR_STRIDE                  19:8

#define LWE397_IDX_ATTRIBUTE_0_ATTR_SIZE                    6:4

#define LWE397_IDX_ATTRIBUTE_0_ATTR_FMT                     3:0
#define LWE397_IDX_ATTRIBUTE_0_ATTR_FMT_U8                        (0)
#define LWE397_IDX_ATTRIBUTE_0_ATTR_FMT_U8N                       (1)
#define LWE397_IDX_ATTRIBUTE_0_ATTR_FMT_S8                        (2)
#define LWE397_IDX_ATTRIBUTE_0_ATTR_FMT_S8N                       (3)
#define LWE397_IDX_ATTRIBUTE_0_ATTR_FMT_U16                       (4)
#define LWE397_IDX_ATTRIBUTE_0_ATTR_FMT_U16N                      (5)
#define LWE397_IDX_ATTRIBUTE_0_ATTR_FMT_S16                       (6)
#define LWE397_IDX_ATTRIBUTE_0_ATTR_FMT_S16N                      (7)
#define LWE397_IDX_ATTRIBUTE_0_ATTR_FMT_U32                       (8)
#define LWE397_IDX_ATTRIBUTE_0_ATTR_FMT_U32N                      (9)
#define LWE397_IDX_ATTRIBUTE_0_ATTR_FMT_S32                       (10)
#define LWE397_IDX_ATTRIBUTE_0_ATTR_FMT_S32N                      (11)
#define LWE397_IDX_ATTRIBUTE_0_ATTR_FMT_X32                       (12)
#define LWE397_IDX_ATTRIBUTE_0_ATTR_FMT_F32                       (13)
#define LWE397_IDX_ATTRIBUTE_0_ATTR_FMT_H16                       (14)


// Register LWE397_IDX_ATTRIBUTE  
#define LWE397_IDX_ATTRIBUTE                      (0x100)
#define LWE397_IDX_ATTRIBUTE_ATTR_BASE                      31:0

#define LWE397_IDX_ATTRIBUTE_SNX_ZERO_PRESERVE                      20:20
#define LWE397_IDX_ATTRIBUTE_SNX_ZERO_PRESERVE_DISABLE                    (0)
#define LWE397_IDX_ATTRIBUTE_SNX_ZERO_PRESERVE_ENABLE                     (1)

#define LWE397_IDX_ATTRIBUTE_ATTR_STRIDE                    19:8

#define LWE397_IDX_ATTRIBUTE_ATTR_SIZE                      6:4

#define LWE397_IDX_ATTRIBUTE_ATTR_FMT                       3:0
#define LWE397_IDX_ATTRIBUTE_ATTR_FMT_U8                  (0)
#define LWE397_IDX_ATTRIBUTE_ATTR_FMT_U8N                 (1)
#define LWE397_IDX_ATTRIBUTE_ATTR_FMT_S8                  (2)
#define LWE397_IDX_ATTRIBUTE_ATTR_FMT_S8N                 (3)
#define LWE397_IDX_ATTRIBUTE_ATTR_FMT_U16                 (4)
#define LWE397_IDX_ATTRIBUTE_ATTR_FMT_U16N                        (5)
#define LWE397_IDX_ATTRIBUTE_ATTR_FMT_S16                 (6)
#define LWE397_IDX_ATTRIBUTE_ATTR_FMT_S16N                        (7)
#define LWE397_IDX_ATTRIBUTE_ATTR_FMT_U32                 (8)
#define LWE397_IDX_ATTRIBUTE_ATTR_FMT_U32N                        (9)
#define LWE397_IDX_ATTRIBUTE_ATTR_FMT_S32                 (10)
#define LWE397_IDX_ATTRIBUTE_ATTR_FMT_S32N                        (11)
#define LWE397_IDX_ATTRIBUTE_ATTR_FMT_X32                 (12)
#define LWE397_IDX_ATTRIBUTE_ATTR_FMT_F32                 (13)
#define LWE397_IDX_ATTRIBUTE_ATTR_FMT_H16                 (14)


// Register LWE397_IDX_ATTRIBUTE_1  
#define LWE397_IDX_ATTRIBUTE_1                    (0x101)
#define LWE397_IDX_ATTRIBUTE_1_ATTR_BASE                    31:0

#define LWE397_IDX_ATTRIBUTE_1_SNX_ZERO_PRESERVE                    20:20
#define LWE397_IDX_ATTRIBUTE_1_SNX_ZERO_PRESERVE_DISABLE                  (0)
#define LWE397_IDX_ATTRIBUTE_1_SNX_ZERO_PRESERVE_ENABLE                   (1)

#define LWE397_IDX_ATTRIBUTE_1_ATTR_STRIDE                  19:8

#define LWE397_IDX_ATTRIBUTE_1_ATTR_SIZE                    6:4

#define LWE397_IDX_ATTRIBUTE_1_ATTR_FMT                     3:0
#define LWE397_IDX_ATTRIBUTE_1_ATTR_FMT_U8                        (0)
#define LWE397_IDX_ATTRIBUTE_1_ATTR_FMT_U8N                       (1)
#define LWE397_IDX_ATTRIBUTE_1_ATTR_FMT_S8                        (2)
#define LWE397_IDX_ATTRIBUTE_1_ATTR_FMT_S8N                       (3)
#define LWE397_IDX_ATTRIBUTE_1_ATTR_FMT_U16                       (4)
#define LWE397_IDX_ATTRIBUTE_1_ATTR_FMT_U16N                      (5)
#define LWE397_IDX_ATTRIBUTE_1_ATTR_FMT_S16                       (6)
#define LWE397_IDX_ATTRIBUTE_1_ATTR_FMT_S16N                      (7)
#define LWE397_IDX_ATTRIBUTE_1_ATTR_FMT_U32                       (8)
#define LWE397_IDX_ATTRIBUTE_1_ATTR_FMT_U32N                      (9)
#define LWE397_IDX_ATTRIBUTE_1_ATTR_FMT_S32                       (10)
#define LWE397_IDX_ATTRIBUTE_1_ATTR_FMT_S32N                      (11)
#define LWE397_IDX_ATTRIBUTE_1_ATTR_FMT_X32                       (12)
#define LWE397_IDX_ATTRIBUTE_1_ATTR_FMT_F32                       (13)
#define LWE397_IDX_ATTRIBUTE_1_ATTR_FMT_H16                       (14)


// Register LWE397_IDX_ATTRIBUTE_2  
#define LWE397_IDX_ATTRIBUTE_2                    (0x102)
#define LWE397_IDX_ATTRIBUTE_2_ATTR_BASE                    31:0

#define LWE397_IDX_ATTRIBUTE_2_SNX_ZERO_PRESERVE                    20:20
#define LWE397_IDX_ATTRIBUTE_2_SNX_ZERO_PRESERVE_DISABLE                  (0)
#define LWE397_IDX_ATTRIBUTE_2_SNX_ZERO_PRESERVE_ENABLE                   (1)

#define LWE397_IDX_ATTRIBUTE_2_ATTR_STRIDE                  19:8

#define LWE397_IDX_ATTRIBUTE_2_ATTR_SIZE                    6:4

#define LWE397_IDX_ATTRIBUTE_2_ATTR_FMT                     3:0
#define LWE397_IDX_ATTRIBUTE_2_ATTR_FMT_U8                        (0)
#define LWE397_IDX_ATTRIBUTE_2_ATTR_FMT_U8N                       (1)
#define LWE397_IDX_ATTRIBUTE_2_ATTR_FMT_S8                        (2)
#define LWE397_IDX_ATTRIBUTE_2_ATTR_FMT_S8N                       (3)
#define LWE397_IDX_ATTRIBUTE_2_ATTR_FMT_U16                       (4)
#define LWE397_IDX_ATTRIBUTE_2_ATTR_FMT_U16N                      (5)
#define LWE397_IDX_ATTRIBUTE_2_ATTR_FMT_S16                       (6)
#define LWE397_IDX_ATTRIBUTE_2_ATTR_FMT_S16N                      (7)
#define LWE397_IDX_ATTRIBUTE_2_ATTR_FMT_U32                       (8)
#define LWE397_IDX_ATTRIBUTE_2_ATTR_FMT_U32N                      (9)
#define LWE397_IDX_ATTRIBUTE_2_ATTR_FMT_S32                       (10)
#define LWE397_IDX_ATTRIBUTE_2_ATTR_FMT_S32N                      (11)
#define LWE397_IDX_ATTRIBUTE_2_ATTR_FMT_X32                       (12)
#define LWE397_IDX_ATTRIBUTE_2_ATTR_FMT_F32                       (13)
#define LWE397_IDX_ATTRIBUTE_2_ATTR_FMT_H16                       (14)


// Register LWE397_IDX_ATTRIBUTE_3  
#define LWE397_IDX_ATTRIBUTE_3                    (0x103)
#define LWE397_IDX_ATTRIBUTE_3_ATTR_BASE                    31:0

#define LWE397_IDX_ATTRIBUTE_3_SNX_ZERO_PRESERVE                    20:20
#define LWE397_IDX_ATTRIBUTE_3_SNX_ZERO_PRESERVE_DISABLE                  (0)
#define LWE397_IDX_ATTRIBUTE_3_SNX_ZERO_PRESERVE_ENABLE                   (1)

#define LWE397_IDX_ATTRIBUTE_3_ATTR_STRIDE                  19:8

#define LWE397_IDX_ATTRIBUTE_3_ATTR_SIZE                    6:4

#define LWE397_IDX_ATTRIBUTE_3_ATTR_FMT                     3:0
#define LWE397_IDX_ATTRIBUTE_3_ATTR_FMT_U8                        (0)
#define LWE397_IDX_ATTRIBUTE_3_ATTR_FMT_U8N                       (1)
#define LWE397_IDX_ATTRIBUTE_3_ATTR_FMT_S8                        (2)
#define LWE397_IDX_ATTRIBUTE_3_ATTR_FMT_S8N                       (3)
#define LWE397_IDX_ATTRIBUTE_3_ATTR_FMT_U16                       (4)
#define LWE397_IDX_ATTRIBUTE_3_ATTR_FMT_U16N                      (5)
#define LWE397_IDX_ATTRIBUTE_3_ATTR_FMT_S16                       (6)
#define LWE397_IDX_ATTRIBUTE_3_ATTR_FMT_S16N                      (7)
#define LWE397_IDX_ATTRIBUTE_3_ATTR_FMT_U32                       (8)
#define LWE397_IDX_ATTRIBUTE_3_ATTR_FMT_U32N                      (9)
#define LWE397_IDX_ATTRIBUTE_3_ATTR_FMT_S32                       (10)
#define LWE397_IDX_ATTRIBUTE_3_ATTR_FMT_S32N                      (11)
#define LWE397_IDX_ATTRIBUTE_3_ATTR_FMT_X32                       (12)
#define LWE397_IDX_ATTRIBUTE_3_ATTR_FMT_F32                       (13)
#define LWE397_IDX_ATTRIBUTE_3_ATTR_FMT_H16                       (14)


// Register LWE397_IDX_ATTRIBUTE_4  
#define LWE397_IDX_ATTRIBUTE_4                    (0x104)
#define LWE397_IDX_ATTRIBUTE_4_ATTR_BASE                    31:0

#define LWE397_IDX_ATTRIBUTE_4_SNX_ZERO_PRESERVE                    20:20
#define LWE397_IDX_ATTRIBUTE_4_SNX_ZERO_PRESERVE_DISABLE                  (0)
#define LWE397_IDX_ATTRIBUTE_4_SNX_ZERO_PRESERVE_ENABLE                   (1)

#define LWE397_IDX_ATTRIBUTE_4_ATTR_STRIDE                  19:8

#define LWE397_IDX_ATTRIBUTE_4_ATTR_SIZE                    6:4

#define LWE397_IDX_ATTRIBUTE_4_ATTR_FMT                     3:0
#define LWE397_IDX_ATTRIBUTE_4_ATTR_FMT_U8                        (0)
#define LWE397_IDX_ATTRIBUTE_4_ATTR_FMT_U8N                       (1)
#define LWE397_IDX_ATTRIBUTE_4_ATTR_FMT_S8                        (2)
#define LWE397_IDX_ATTRIBUTE_4_ATTR_FMT_S8N                       (3)
#define LWE397_IDX_ATTRIBUTE_4_ATTR_FMT_U16                       (4)
#define LWE397_IDX_ATTRIBUTE_4_ATTR_FMT_U16N                      (5)
#define LWE397_IDX_ATTRIBUTE_4_ATTR_FMT_S16                       (6)
#define LWE397_IDX_ATTRIBUTE_4_ATTR_FMT_S16N                      (7)
#define LWE397_IDX_ATTRIBUTE_4_ATTR_FMT_U32                       (8)
#define LWE397_IDX_ATTRIBUTE_4_ATTR_FMT_U32N                      (9)
#define LWE397_IDX_ATTRIBUTE_4_ATTR_FMT_S32                       (10)
#define LWE397_IDX_ATTRIBUTE_4_ATTR_FMT_S32N                      (11)
#define LWE397_IDX_ATTRIBUTE_4_ATTR_FMT_X32                       (12)
#define LWE397_IDX_ATTRIBUTE_4_ATTR_FMT_F32                       (13)
#define LWE397_IDX_ATTRIBUTE_4_ATTR_FMT_H16                       (14)


// Register LWE397_IDX_ATTRIBUTE_5  
#define LWE397_IDX_ATTRIBUTE_5                    (0x105)
#define LWE397_IDX_ATTRIBUTE_5_ATTR_BASE                    31:0

#define LWE397_IDX_ATTRIBUTE_5_SNX_ZERO_PRESERVE                    20:20
#define LWE397_IDX_ATTRIBUTE_5_SNX_ZERO_PRESERVE_DISABLE                  (0)
#define LWE397_IDX_ATTRIBUTE_5_SNX_ZERO_PRESERVE_ENABLE                   (1)

#define LWE397_IDX_ATTRIBUTE_5_ATTR_STRIDE                  19:8

#define LWE397_IDX_ATTRIBUTE_5_ATTR_SIZE                    6:4

#define LWE397_IDX_ATTRIBUTE_5_ATTR_FMT                     3:0
#define LWE397_IDX_ATTRIBUTE_5_ATTR_FMT_U8                        (0)
#define LWE397_IDX_ATTRIBUTE_5_ATTR_FMT_U8N                       (1)
#define LWE397_IDX_ATTRIBUTE_5_ATTR_FMT_S8                        (2)
#define LWE397_IDX_ATTRIBUTE_5_ATTR_FMT_S8N                       (3)
#define LWE397_IDX_ATTRIBUTE_5_ATTR_FMT_U16                       (4)
#define LWE397_IDX_ATTRIBUTE_5_ATTR_FMT_U16N                      (5)
#define LWE397_IDX_ATTRIBUTE_5_ATTR_FMT_S16                       (6)
#define LWE397_IDX_ATTRIBUTE_5_ATTR_FMT_S16N                      (7)
#define LWE397_IDX_ATTRIBUTE_5_ATTR_FMT_U32                       (8)
#define LWE397_IDX_ATTRIBUTE_5_ATTR_FMT_U32N                      (9)
#define LWE397_IDX_ATTRIBUTE_5_ATTR_FMT_S32                       (10)
#define LWE397_IDX_ATTRIBUTE_5_ATTR_FMT_S32N                      (11)
#define LWE397_IDX_ATTRIBUTE_5_ATTR_FMT_X32                       (12)
#define LWE397_IDX_ATTRIBUTE_5_ATTR_FMT_F32                       (13)
#define LWE397_IDX_ATTRIBUTE_5_ATTR_FMT_H16                       (14)


// Register LWE397_IDX_ATTRIBUTE_6  
#define LWE397_IDX_ATTRIBUTE_6                    (0x106)
#define LWE397_IDX_ATTRIBUTE_6_ATTR_BASE                    31:0

#define LWE397_IDX_ATTRIBUTE_6_SNX_ZERO_PRESERVE                    20:20
#define LWE397_IDX_ATTRIBUTE_6_SNX_ZERO_PRESERVE_DISABLE                  (0)
#define LWE397_IDX_ATTRIBUTE_6_SNX_ZERO_PRESERVE_ENABLE                   (1)

#define LWE397_IDX_ATTRIBUTE_6_ATTR_STRIDE                  19:8

#define LWE397_IDX_ATTRIBUTE_6_ATTR_SIZE                    6:4

#define LWE397_IDX_ATTRIBUTE_6_ATTR_FMT                     3:0
#define LWE397_IDX_ATTRIBUTE_6_ATTR_FMT_U8                        (0)
#define LWE397_IDX_ATTRIBUTE_6_ATTR_FMT_U8N                       (1)
#define LWE397_IDX_ATTRIBUTE_6_ATTR_FMT_S8                        (2)
#define LWE397_IDX_ATTRIBUTE_6_ATTR_FMT_S8N                       (3)
#define LWE397_IDX_ATTRIBUTE_6_ATTR_FMT_U16                       (4)
#define LWE397_IDX_ATTRIBUTE_6_ATTR_FMT_U16N                      (5)
#define LWE397_IDX_ATTRIBUTE_6_ATTR_FMT_S16                       (6)
#define LWE397_IDX_ATTRIBUTE_6_ATTR_FMT_S16N                      (7)
#define LWE397_IDX_ATTRIBUTE_6_ATTR_FMT_U32                       (8)
#define LWE397_IDX_ATTRIBUTE_6_ATTR_FMT_U32N                      (9)
#define LWE397_IDX_ATTRIBUTE_6_ATTR_FMT_S32                       (10)
#define LWE397_IDX_ATTRIBUTE_6_ATTR_FMT_S32N                      (11)
#define LWE397_IDX_ATTRIBUTE_6_ATTR_FMT_X32                       (12)
#define LWE397_IDX_ATTRIBUTE_6_ATTR_FMT_F32                       (13)
#define LWE397_IDX_ATTRIBUTE_6_ATTR_FMT_H16                       (14)


// Register LWE397_IDX_ATTRIBUTE_7  
#define LWE397_IDX_ATTRIBUTE_7                    (0x107)
#define LWE397_IDX_ATTRIBUTE_7_ATTR_BASE                    31:0

#define LWE397_IDX_ATTRIBUTE_7_SNX_ZERO_PRESERVE                    20:20
#define LWE397_IDX_ATTRIBUTE_7_SNX_ZERO_PRESERVE_DISABLE                  (0)
#define LWE397_IDX_ATTRIBUTE_7_SNX_ZERO_PRESERVE_ENABLE                   (1)

#define LWE397_IDX_ATTRIBUTE_7_ATTR_STRIDE                  19:8

#define LWE397_IDX_ATTRIBUTE_7_ATTR_SIZE                    6:4

#define LWE397_IDX_ATTRIBUTE_7_ATTR_FMT                     3:0
#define LWE397_IDX_ATTRIBUTE_7_ATTR_FMT_U8                        (0)
#define LWE397_IDX_ATTRIBUTE_7_ATTR_FMT_U8N                       (1)
#define LWE397_IDX_ATTRIBUTE_7_ATTR_FMT_S8                        (2)
#define LWE397_IDX_ATTRIBUTE_7_ATTR_FMT_S8N                       (3)
#define LWE397_IDX_ATTRIBUTE_7_ATTR_FMT_U16                       (4)
#define LWE397_IDX_ATTRIBUTE_7_ATTR_FMT_U16N                      (5)
#define LWE397_IDX_ATTRIBUTE_7_ATTR_FMT_S16                       (6)
#define LWE397_IDX_ATTRIBUTE_7_ATTR_FMT_S16N                      (7)
#define LWE397_IDX_ATTRIBUTE_7_ATTR_FMT_U32                       (8)
#define LWE397_IDX_ATTRIBUTE_7_ATTR_FMT_U32N                      (9)
#define LWE397_IDX_ATTRIBUTE_7_ATTR_FMT_S32                       (10)
#define LWE397_IDX_ATTRIBUTE_7_ATTR_FMT_S32N                      (11)
#define LWE397_IDX_ATTRIBUTE_7_ATTR_FMT_X32                       (12)
#define LWE397_IDX_ATTRIBUTE_7_ATTR_FMT_F32                       (13)
#define LWE397_IDX_ATTRIBUTE_7_ATTR_FMT_H16                       (14)


// Register LWE397_IDX_ATTRIBUTE_8  
#define LWE397_IDX_ATTRIBUTE_8                    (0x108)
#define LWE397_IDX_ATTRIBUTE_8_ATTR_BASE                    31:0

#define LWE397_IDX_ATTRIBUTE_8_SNX_ZERO_PRESERVE                    20:20
#define LWE397_IDX_ATTRIBUTE_8_SNX_ZERO_PRESERVE_DISABLE                  (0)
#define LWE397_IDX_ATTRIBUTE_8_SNX_ZERO_PRESERVE_ENABLE                   (1)

#define LWE397_IDX_ATTRIBUTE_8_ATTR_STRIDE                  19:8

#define LWE397_IDX_ATTRIBUTE_8_ATTR_SIZE                    6:4

#define LWE397_IDX_ATTRIBUTE_8_ATTR_FMT                     3:0
#define LWE397_IDX_ATTRIBUTE_8_ATTR_FMT_U8                        (0)
#define LWE397_IDX_ATTRIBUTE_8_ATTR_FMT_U8N                       (1)
#define LWE397_IDX_ATTRIBUTE_8_ATTR_FMT_S8                        (2)
#define LWE397_IDX_ATTRIBUTE_8_ATTR_FMT_S8N                       (3)
#define LWE397_IDX_ATTRIBUTE_8_ATTR_FMT_U16                       (4)
#define LWE397_IDX_ATTRIBUTE_8_ATTR_FMT_U16N                      (5)
#define LWE397_IDX_ATTRIBUTE_8_ATTR_FMT_S16                       (6)
#define LWE397_IDX_ATTRIBUTE_8_ATTR_FMT_S16N                      (7)
#define LWE397_IDX_ATTRIBUTE_8_ATTR_FMT_U32                       (8)
#define LWE397_IDX_ATTRIBUTE_8_ATTR_FMT_U32N                      (9)
#define LWE397_IDX_ATTRIBUTE_8_ATTR_FMT_S32                       (10)
#define LWE397_IDX_ATTRIBUTE_8_ATTR_FMT_S32N                      (11)
#define LWE397_IDX_ATTRIBUTE_8_ATTR_FMT_X32                       (12)
#define LWE397_IDX_ATTRIBUTE_8_ATTR_FMT_F32                       (13)
#define LWE397_IDX_ATTRIBUTE_8_ATTR_FMT_H16                       (14)


// Register LWE397_IDX_ATTRIBUTE_9  
#define LWE397_IDX_ATTRIBUTE_9                    (0x109)
#define LWE397_IDX_ATTRIBUTE_9_ATTR_BASE                    31:0

#define LWE397_IDX_ATTRIBUTE_9_SNX_ZERO_PRESERVE                    20:20
#define LWE397_IDX_ATTRIBUTE_9_SNX_ZERO_PRESERVE_DISABLE                  (0)
#define LWE397_IDX_ATTRIBUTE_9_SNX_ZERO_PRESERVE_ENABLE                   (1)

#define LWE397_IDX_ATTRIBUTE_9_ATTR_STRIDE                  19:8

#define LWE397_IDX_ATTRIBUTE_9_ATTR_SIZE                    6:4

#define LWE397_IDX_ATTRIBUTE_9_ATTR_FMT                     3:0
#define LWE397_IDX_ATTRIBUTE_9_ATTR_FMT_U8                        (0)
#define LWE397_IDX_ATTRIBUTE_9_ATTR_FMT_U8N                       (1)
#define LWE397_IDX_ATTRIBUTE_9_ATTR_FMT_S8                        (2)
#define LWE397_IDX_ATTRIBUTE_9_ATTR_FMT_S8N                       (3)
#define LWE397_IDX_ATTRIBUTE_9_ATTR_FMT_U16                       (4)
#define LWE397_IDX_ATTRIBUTE_9_ATTR_FMT_U16N                      (5)
#define LWE397_IDX_ATTRIBUTE_9_ATTR_FMT_S16                       (6)
#define LWE397_IDX_ATTRIBUTE_9_ATTR_FMT_S16N                      (7)
#define LWE397_IDX_ATTRIBUTE_9_ATTR_FMT_U32                       (8)
#define LWE397_IDX_ATTRIBUTE_9_ATTR_FMT_U32N                      (9)
#define LWE397_IDX_ATTRIBUTE_9_ATTR_FMT_S32                       (10)
#define LWE397_IDX_ATTRIBUTE_9_ATTR_FMT_S32N                      (11)
#define LWE397_IDX_ATTRIBUTE_9_ATTR_FMT_X32                       (12)
#define LWE397_IDX_ATTRIBUTE_9_ATTR_FMT_F32                       (13)
#define LWE397_IDX_ATTRIBUTE_9_ATTR_FMT_H16                       (14)


// Register LWE397_IDX_ATTRIBUTE_10  
#define LWE397_IDX_ATTRIBUTE_10                   (0x10a)
#define LWE397_IDX_ATTRIBUTE_10_ATTR_BASE                   31:0

#define LWE397_IDX_ATTRIBUTE_10_SNX_ZERO_PRESERVE                   20:20
#define LWE397_IDX_ATTRIBUTE_10_SNX_ZERO_PRESERVE_DISABLE                 (0)
#define LWE397_IDX_ATTRIBUTE_10_SNX_ZERO_PRESERVE_ENABLE                  (1)

#define LWE397_IDX_ATTRIBUTE_10_ATTR_STRIDE                 19:8

#define LWE397_IDX_ATTRIBUTE_10_ATTR_SIZE                   6:4

#define LWE397_IDX_ATTRIBUTE_10_ATTR_FMT                    3:0
#define LWE397_IDX_ATTRIBUTE_10_ATTR_FMT_U8                       (0)
#define LWE397_IDX_ATTRIBUTE_10_ATTR_FMT_U8N                      (1)
#define LWE397_IDX_ATTRIBUTE_10_ATTR_FMT_S8                       (2)
#define LWE397_IDX_ATTRIBUTE_10_ATTR_FMT_S8N                      (3)
#define LWE397_IDX_ATTRIBUTE_10_ATTR_FMT_U16                      (4)
#define LWE397_IDX_ATTRIBUTE_10_ATTR_FMT_U16N                     (5)
#define LWE397_IDX_ATTRIBUTE_10_ATTR_FMT_S16                      (6)
#define LWE397_IDX_ATTRIBUTE_10_ATTR_FMT_S16N                     (7)
#define LWE397_IDX_ATTRIBUTE_10_ATTR_FMT_U32                      (8)
#define LWE397_IDX_ATTRIBUTE_10_ATTR_FMT_U32N                     (9)
#define LWE397_IDX_ATTRIBUTE_10_ATTR_FMT_S32                      (10)
#define LWE397_IDX_ATTRIBUTE_10_ATTR_FMT_S32N                     (11)
#define LWE397_IDX_ATTRIBUTE_10_ATTR_FMT_X32                      (12)
#define LWE397_IDX_ATTRIBUTE_10_ATTR_FMT_F32                      (13)
#define LWE397_IDX_ATTRIBUTE_10_ATTR_FMT_H16                      (14)


// Register LWE397_IDX_ATTRIBUTE_11  
#define LWE397_IDX_ATTRIBUTE_11                   (0x10b)
#define LWE397_IDX_ATTRIBUTE_11_ATTR_BASE                   31:0

#define LWE397_IDX_ATTRIBUTE_11_SNX_ZERO_PRESERVE                   20:20
#define LWE397_IDX_ATTRIBUTE_11_SNX_ZERO_PRESERVE_DISABLE                 (0)
#define LWE397_IDX_ATTRIBUTE_11_SNX_ZERO_PRESERVE_ENABLE                  (1)

#define LWE397_IDX_ATTRIBUTE_11_ATTR_STRIDE                 19:8

#define LWE397_IDX_ATTRIBUTE_11_ATTR_SIZE                   6:4

#define LWE397_IDX_ATTRIBUTE_11_ATTR_FMT                    3:0
#define LWE397_IDX_ATTRIBUTE_11_ATTR_FMT_U8                       (0)
#define LWE397_IDX_ATTRIBUTE_11_ATTR_FMT_U8N                      (1)
#define LWE397_IDX_ATTRIBUTE_11_ATTR_FMT_S8                       (2)
#define LWE397_IDX_ATTRIBUTE_11_ATTR_FMT_S8N                      (3)
#define LWE397_IDX_ATTRIBUTE_11_ATTR_FMT_U16                      (4)
#define LWE397_IDX_ATTRIBUTE_11_ATTR_FMT_U16N                     (5)
#define LWE397_IDX_ATTRIBUTE_11_ATTR_FMT_S16                      (6)
#define LWE397_IDX_ATTRIBUTE_11_ATTR_FMT_S16N                     (7)
#define LWE397_IDX_ATTRIBUTE_11_ATTR_FMT_U32                      (8)
#define LWE397_IDX_ATTRIBUTE_11_ATTR_FMT_U32N                     (9)
#define LWE397_IDX_ATTRIBUTE_11_ATTR_FMT_S32                      (10)
#define LWE397_IDX_ATTRIBUTE_11_ATTR_FMT_S32N                     (11)
#define LWE397_IDX_ATTRIBUTE_11_ATTR_FMT_X32                      (12)
#define LWE397_IDX_ATTRIBUTE_11_ATTR_FMT_F32                      (13)
#define LWE397_IDX_ATTRIBUTE_11_ATTR_FMT_H16                      (14)


// Register LWE397_IDX_ATTRIBUTE_12  
#define LWE397_IDX_ATTRIBUTE_12                   (0x10c)
#define LWE397_IDX_ATTRIBUTE_12_ATTR_BASE                   31:0

#define LWE397_IDX_ATTRIBUTE_12_SNX_ZERO_PRESERVE                   20:20
#define LWE397_IDX_ATTRIBUTE_12_SNX_ZERO_PRESERVE_DISABLE                 (0)
#define LWE397_IDX_ATTRIBUTE_12_SNX_ZERO_PRESERVE_ENABLE                  (1)

#define LWE397_IDX_ATTRIBUTE_12_ATTR_STRIDE                 19:8

#define LWE397_IDX_ATTRIBUTE_12_ATTR_SIZE                   6:4

#define LWE397_IDX_ATTRIBUTE_12_ATTR_FMT                    3:0
#define LWE397_IDX_ATTRIBUTE_12_ATTR_FMT_U8                       (0)
#define LWE397_IDX_ATTRIBUTE_12_ATTR_FMT_U8N                      (1)
#define LWE397_IDX_ATTRIBUTE_12_ATTR_FMT_S8                       (2)
#define LWE397_IDX_ATTRIBUTE_12_ATTR_FMT_S8N                      (3)
#define LWE397_IDX_ATTRIBUTE_12_ATTR_FMT_U16                      (4)
#define LWE397_IDX_ATTRIBUTE_12_ATTR_FMT_U16N                     (5)
#define LWE397_IDX_ATTRIBUTE_12_ATTR_FMT_S16                      (6)
#define LWE397_IDX_ATTRIBUTE_12_ATTR_FMT_S16N                     (7)
#define LWE397_IDX_ATTRIBUTE_12_ATTR_FMT_U32                      (8)
#define LWE397_IDX_ATTRIBUTE_12_ATTR_FMT_U32N                     (9)
#define LWE397_IDX_ATTRIBUTE_12_ATTR_FMT_S32                      (10)
#define LWE397_IDX_ATTRIBUTE_12_ATTR_FMT_S32N                     (11)
#define LWE397_IDX_ATTRIBUTE_12_ATTR_FMT_X32                      (12)
#define LWE397_IDX_ATTRIBUTE_12_ATTR_FMT_F32                      (13)
#define LWE397_IDX_ATTRIBUTE_12_ATTR_FMT_H16                      (14)


// Register LWE397_IDX_ATTRIBUTE_13  
#define LWE397_IDX_ATTRIBUTE_13                   (0x10d)
#define LWE397_IDX_ATTRIBUTE_13_ATTR_BASE                   31:0

#define LWE397_IDX_ATTRIBUTE_13_SNX_ZERO_PRESERVE                   20:20
#define LWE397_IDX_ATTRIBUTE_13_SNX_ZERO_PRESERVE_DISABLE                 (0)
#define LWE397_IDX_ATTRIBUTE_13_SNX_ZERO_PRESERVE_ENABLE                  (1)

#define LWE397_IDX_ATTRIBUTE_13_ATTR_STRIDE                 19:8

#define LWE397_IDX_ATTRIBUTE_13_ATTR_SIZE                   6:4

#define LWE397_IDX_ATTRIBUTE_13_ATTR_FMT                    3:0
#define LWE397_IDX_ATTRIBUTE_13_ATTR_FMT_U8                       (0)
#define LWE397_IDX_ATTRIBUTE_13_ATTR_FMT_U8N                      (1)
#define LWE397_IDX_ATTRIBUTE_13_ATTR_FMT_S8                       (2)
#define LWE397_IDX_ATTRIBUTE_13_ATTR_FMT_S8N                      (3)
#define LWE397_IDX_ATTRIBUTE_13_ATTR_FMT_U16                      (4)
#define LWE397_IDX_ATTRIBUTE_13_ATTR_FMT_U16N                     (5)
#define LWE397_IDX_ATTRIBUTE_13_ATTR_FMT_S16                      (6)
#define LWE397_IDX_ATTRIBUTE_13_ATTR_FMT_S16N                     (7)
#define LWE397_IDX_ATTRIBUTE_13_ATTR_FMT_U32                      (8)
#define LWE397_IDX_ATTRIBUTE_13_ATTR_FMT_U32N                     (9)
#define LWE397_IDX_ATTRIBUTE_13_ATTR_FMT_S32                      (10)
#define LWE397_IDX_ATTRIBUTE_13_ATTR_FMT_S32N                     (11)
#define LWE397_IDX_ATTRIBUTE_13_ATTR_FMT_X32                      (12)
#define LWE397_IDX_ATTRIBUTE_13_ATTR_FMT_F32                      (13)
#define LWE397_IDX_ATTRIBUTE_13_ATTR_FMT_H16                      (14)


// Register LWE397_IDX_ATTRIBUTE_14  
#define LWE397_IDX_ATTRIBUTE_14                   (0x10e)
#define LWE397_IDX_ATTRIBUTE_14_ATTR_BASE                   31:0

#define LWE397_IDX_ATTRIBUTE_14_SNX_ZERO_PRESERVE                   20:20
#define LWE397_IDX_ATTRIBUTE_14_SNX_ZERO_PRESERVE_DISABLE                 (0)
#define LWE397_IDX_ATTRIBUTE_14_SNX_ZERO_PRESERVE_ENABLE                  (1)

#define LWE397_IDX_ATTRIBUTE_14_ATTR_STRIDE                 19:8

#define LWE397_IDX_ATTRIBUTE_14_ATTR_SIZE                   6:4

#define LWE397_IDX_ATTRIBUTE_14_ATTR_FMT                    3:0
#define LWE397_IDX_ATTRIBUTE_14_ATTR_FMT_U8                       (0)
#define LWE397_IDX_ATTRIBUTE_14_ATTR_FMT_U8N                      (1)
#define LWE397_IDX_ATTRIBUTE_14_ATTR_FMT_S8                       (2)
#define LWE397_IDX_ATTRIBUTE_14_ATTR_FMT_S8N                      (3)
#define LWE397_IDX_ATTRIBUTE_14_ATTR_FMT_U16                      (4)
#define LWE397_IDX_ATTRIBUTE_14_ATTR_FMT_U16N                     (5)
#define LWE397_IDX_ATTRIBUTE_14_ATTR_FMT_S16                      (6)
#define LWE397_IDX_ATTRIBUTE_14_ATTR_FMT_S16N                     (7)
#define LWE397_IDX_ATTRIBUTE_14_ATTR_FMT_U32                      (8)
#define LWE397_IDX_ATTRIBUTE_14_ATTR_FMT_U32N                     (9)
#define LWE397_IDX_ATTRIBUTE_14_ATTR_FMT_S32                      (10)
#define LWE397_IDX_ATTRIBUTE_14_ATTR_FMT_S32N                     (11)
#define LWE397_IDX_ATTRIBUTE_14_ATTR_FMT_X32                      (12)
#define LWE397_IDX_ATTRIBUTE_14_ATTR_FMT_F32                      (13)
#define LWE397_IDX_ATTRIBUTE_14_ATTR_FMT_H16                      (14)


// Register LWE397_IDX_ATTRIBUTE_15  
#define LWE397_IDX_ATTRIBUTE_15                   (0x10f)
#define LWE397_IDX_ATTRIBUTE_15_ATTR_BASE                   31:0

#define LWE397_IDX_ATTRIBUTE_15_SNX_ZERO_PRESERVE                   20:20
#define LWE397_IDX_ATTRIBUTE_15_SNX_ZERO_PRESERVE_DISABLE                 (0)
#define LWE397_IDX_ATTRIBUTE_15_SNX_ZERO_PRESERVE_ENABLE                  (1)

#define LWE397_IDX_ATTRIBUTE_15_ATTR_STRIDE                 19:8

#define LWE397_IDX_ATTRIBUTE_15_ATTR_SIZE                   6:4

#define LWE397_IDX_ATTRIBUTE_15_ATTR_FMT                    3:0
#define LWE397_IDX_ATTRIBUTE_15_ATTR_FMT_U8                       (0)
#define LWE397_IDX_ATTRIBUTE_15_ATTR_FMT_U8N                      (1)
#define LWE397_IDX_ATTRIBUTE_15_ATTR_FMT_S8                       (2)
#define LWE397_IDX_ATTRIBUTE_15_ATTR_FMT_S8N                      (3)
#define LWE397_IDX_ATTRIBUTE_15_ATTR_FMT_U16                      (4)
#define LWE397_IDX_ATTRIBUTE_15_ATTR_FMT_U16N                     (5)
#define LWE397_IDX_ATTRIBUTE_15_ATTR_FMT_S16                      (6)
#define LWE397_IDX_ATTRIBUTE_15_ATTR_FMT_S16N                     (7)
#define LWE397_IDX_ATTRIBUTE_15_ATTR_FMT_U32                      (8)
#define LWE397_IDX_ATTRIBUTE_15_ATTR_FMT_U32N                     (9)
#define LWE397_IDX_ATTRIBUTE_15_ATTR_FMT_S32                      (10)
#define LWE397_IDX_ATTRIBUTE_15_ATTR_FMT_S32N                     (11)
#define LWE397_IDX_ATTRIBUTE_15_ATTR_FMT_X32                      (12)
#define LWE397_IDX_ATTRIBUTE_15_ATTR_FMT_F32                      (13)
#define LWE397_IDX_ATTRIBUTE_15_ATTR_FMT_H16                      (14)


// Register LWE397_IDX_ATTRIBUTE_16  
#define LWE397_IDX_ATTRIBUTE_16                   (0x110)
#define LWE397_IDX_ATTRIBUTE_16_ATTR_BASE                   31:0

#define LWE397_IDX_ATTRIBUTE_16_SNX_ZERO_PRESERVE                   20:20
#define LWE397_IDX_ATTRIBUTE_16_SNX_ZERO_PRESERVE_DISABLE                 (0)
#define LWE397_IDX_ATTRIBUTE_16_SNX_ZERO_PRESERVE_ENABLE                  (1)

#define LWE397_IDX_ATTRIBUTE_16_ATTR_STRIDE                 19:8

#define LWE397_IDX_ATTRIBUTE_16_ATTR_SIZE                   6:4

#define LWE397_IDX_ATTRIBUTE_16_ATTR_FMT                    3:0
#define LWE397_IDX_ATTRIBUTE_16_ATTR_FMT_U8                       (0)
#define LWE397_IDX_ATTRIBUTE_16_ATTR_FMT_U8N                      (1)
#define LWE397_IDX_ATTRIBUTE_16_ATTR_FMT_S8                       (2)
#define LWE397_IDX_ATTRIBUTE_16_ATTR_FMT_S8N                      (3)
#define LWE397_IDX_ATTRIBUTE_16_ATTR_FMT_U16                      (4)
#define LWE397_IDX_ATTRIBUTE_16_ATTR_FMT_U16N                     (5)
#define LWE397_IDX_ATTRIBUTE_16_ATTR_FMT_S16                      (6)
#define LWE397_IDX_ATTRIBUTE_16_ATTR_FMT_S16N                     (7)
#define LWE397_IDX_ATTRIBUTE_16_ATTR_FMT_U32                      (8)
#define LWE397_IDX_ATTRIBUTE_16_ATTR_FMT_U32N                     (9)
#define LWE397_IDX_ATTRIBUTE_16_ATTR_FMT_S32                      (10)
#define LWE397_IDX_ATTRIBUTE_16_ATTR_FMT_S32N                     (11)
#define LWE397_IDX_ATTRIBUTE_16_ATTR_FMT_X32                      (12)
#define LWE397_IDX_ATTRIBUTE_16_ATTR_FMT_F32                      (13)
#define LWE397_IDX_ATTRIBUTE_16_ATTR_FMT_H16                      (14)


// Register LWE397_IDX_ATTRIBUTE_17  
#define LWE397_IDX_ATTRIBUTE_17                   (0x111)
#define LWE397_IDX_ATTRIBUTE_17_ATTR_BASE                   31:0

#define LWE397_IDX_ATTRIBUTE_17_SNX_ZERO_PRESERVE                   20:20
#define LWE397_IDX_ATTRIBUTE_17_SNX_ZERO_PRESERVE_DISABLE                 (0)
#define LWE397_IDX_ATTRIBUTE_17_SNX_ZERO_PRESERVE_ENABLE                  (1)

#define LWE397_IDX_ATTRIBUTE_17_ATTR_STRIDE                 19:8

#define LWE397_IDX_ATTRIBUTE_17_ATTR_SIZE                   6:4

#define LWE397_IDX_ATTRIBUTE_17_ATTR_FMT                    3:0
#define LWE397_IDX_ATTRIBUTE_17_ATTR_FMT_U8                       (0)
#define LWE397_IDX_ATTRIBUTE_17_ATTR_FMT_U8N                      (1)
#define LWE397_IDX_ATTRIBUTE_17_ATTR_FMT_S8                       (2)
#define LWE397_IDX_ATTRIBUTE_17_ATTR_FMT_S8N                      (3)
#define LWE397_IDX_ATTRIBUTE_17_ATTR_FMT_U16                      (4)
#define LWE397_IDX_ATTRIBUTE_17_ATTR_FMT_U16N                     (5)
#define LWE397_IDX_ATTRIBUTE_17_ATTR_FMT_S16                      (6)
#define LWE397_IDX_ATTRIBUTE_17_ATTR_FMT_S16N                     (7)
#define LWE397_IDX_ATTRIBUTE_17_ATTR_FMT_U32                      (8)
#define LWE397_IDX_ATTRIBUTE_17_ATTR_FMT_U32N                     (9)
#define LWE397_IDX_ATTRIBUTE_17_ATTR_FMT_S32                      (10)
#define LWE397_IDX_ATTRIBUTE_17_ATTR_FMT_S32N                     (11)
#define LWE397_IDX_ATTRIBUTE_17_ATTR_FMT_X32                      (12)
#define LWE397_IDX_ATTRIBUTE_17_ATTR_FMT_F32                      (13)
#define LWE397_IDX_ATTRIBUTE_17_ATTR_FMT_H16                      (14)


// Register LWE397_IDX_ATTRIBUTE_18  
#define LWE397_IDX_ATTRIBUTE_18                   (0x112)
#define LWE397_IDX_ATTRIBUTE_18_ATTR_BASE                   31:0

#define LWE397_IDX_ATTRIBUTE_18_SNX_ZERO_PRESERVE                   20:20
#define LWE397_IDX_ATTRIBUTE_18_SNX_ZERO_PRESERVE_DISABLE                 (0)
#define LWE397_IDX_ATTRIBUTE_18_SNX_ZERO_PRESERVE_ENABLE                  (1)

#define LWE397_IDX_ATTRIBUTE_18_ATTR_STRIDE                 19:8

#define LWE397_IDX_ATTRIBUTE_18_ATTR_SIZE                   6:4

#define LWE397_IDX_ATTRIBUTE_18_ATTR_FMT                    3:0
#define LWE397_IDX_ATTRIBUTE_18_ATTR_FMT_U8                       (0)
#define LWE397_IDX_ATTRIBUTE_18_ATTR_FMT_U8N                      (1)
#define LWE397_IDX_ATTRIBUTE_18_ATTR_FMT_S8                       (2)
#define LWE397_IDX_ATTRIBUTE_18_ATTR_FMT_S8N                      (3)
#define LWE397_IDX_ATTRIBUTE_18_ATTR_FMT_U16                      (4)
#define LWE397_IDX_ATTRIBUTE_18_ATTR_FMT_U16N                     (5)
#define LWE397_IDX_ATTRIBUTE_18_ATTR_FMT_S16                      (6)
#define LWE397_IDX_ATTRIBUTE_18_ATTR_FMT_S16N                     (7)
#define LWE397_IDX_ATTRIBUTE_18_ATTR_FMT_U32                      (8)
#define LWE397_IDX_ATTRIBUTE_18_ATTR_FMT_U32N                     (9)
#define LWE397_IDX_ATTRIBUTE_18_ATTR_FMT_S32                      (10)
#define LWE397_IDX_ATTRIBUTE_18_ATTR_FMT_S32N                     (11)
#define LWE397_IDX_ATTRIBUTE_18_ATTR_FMT_X32                      (12)
#define LWE397_IDX_ATTRIBUTE_18_ATTR_FMT_F32                      (13)
#define LWE397_IDX_ATTRIBUTE_18_ATTR_FMT_H16                      (14)


// Register LWE397_IDX_ATTRIBUTE_19  
#define LWE397_IDX_ATTRIBUTE_19                   (0x113)
#define LWE397_IDX_ATTRIBUTE_19_ATTR_BASE                   31:0

#define LWE397_IDX_ATTRIBUTE_19_SNX_ZERO_PRESERVE                   20:20
#define LWE397_IDX_ATTRIBUTE_19_SNX_ZERO_PRESERVE_DISABLE                 (0)
#define LWE397_IDX_ATTRIBUTE_19_SNX_ZERO_PRESERVE_ENABLE                  (1)

#define LWE397_IDX_ATTRIBUTE_19_ATTR_STRIDE                 19:8

#define LWE397_IDX_ATTRIBUTE_19_ATTR_SIZE                   6:4

#define LWE397_IDX_ATTRIBUTE_19_ATTR_FMT                    3:0
#define LWE397_IDX_ATTRIBUTE_19_ATTR_FMT_U8                       (0)
#define LWE397_IDX_ATTRIBUTE_19_ATTR_FMT_U8N                      (1)
#define LWE397_IDX_ATTRIBUTE_19_ATTR_FMT_S8                       (2)
#define LWE397_IDX_ATTRIBUTE_19_ATTR_FMT_S8N                      (3)
#define LWE397_IDX_ATTRIBUTE_19_ATTR_FMT_U16                      (4)
#define LWE397_IDX_ATTRIBUTE_19_ATTR_FMT_U16N                     (5)
#define LWE397_IDX_ATTRIBUTE_19_ATTR_FMT_S16                      (6)
#define LWE397_IDX_ATTRIBUTE_19_ATTR_FMT_S16N                     (7)
#define LWE397_IDX_ATTRIBUTE_19_ATTR_FMT_U32                      (8)
#define LWE397_IDX_ATTRIBUTE_19_ATTR_FMT_U32N                     (9)
#define LWE397_IDX_ATTRIBUTE_19_ATTR_FMT_S32                      (10)
#define LWE397_IDX_ATTRIBUTE_19_ATTR_FMT_S32N                     (11)
#define LWE397_IDX_ATTRIBUTE_19_ATTR_FMT_X32                      (12)
#define LWE397_IDX_ATTRIBUTE_19_ATTR_FMT_F32                      (13)
#define LWE397_IDX_ATTRIBUTE_19_ATTR_FMT_H16                      (14)


// Register LWE397_IDX_ATTRIBUTE_20  
#define LWE397_IDX_ATTRIBUTE_20                   (0x114)
#define LWE397_IDX_ATTRIBUTE_20_ATTR_BASE                   31:0

#define LWE397_IDX_ATTRIBUTE_20_SNX_ZERO_PRESERVE                   20:20
#define LWE397_IDX_ATTRIBUTE_20_SNX_ZERO_PRESERVE_DISABLE                 (0)
#define LWE397_IDX_ATTRIBUTE_20_SNX_ZERO_PRESERVE_ENABLE                  (1)

#define LWE397_IDX_ATTRIBUTE_20_ATTR_STRIDE                 19:8

#define LWE397_IDX_ATTRIBUTE_20_ATTR_SIZE                   6:4

#define LWE397_IDX_ATTRIBUTE_20_ATTR_FMT                    3:0
#define LWE397_IDX_ATTRIBUTE_20_ATTR_FMT_U8                       (0)
#define LWE397_IDX_ATTRIBUTE_20_ATTR_FMT_U8N                      (1)
#define LWE397_IDX_ATTRIBUTE_20_ATTR_FMT_S8                       (2)
#define LWE397_IDX_ATTRIBUTE_20_ATTR_FMT_S8N                      (3)
#define LWE397_IDX_ATTRIBUTE_20_ATTR_FMT_U16                      (4)
#define LWE397_IDX_ATTRIBUTE_20_ATTR_FMT_U16N                     (5)
#define LWE397_IDX_ATTRIBUTE_20_ATTR_FMT_S16                      (6)
#define LWE397_IDX_ATTRIBUTE_20_ATTR_FMT_S16N                     (7)
#define LWE397_IDX_ATTRIBUTE_20_ATTR_FMT_U32                      (8)
#define LWE397_IDX_ATTRIBUTE_20_ATTR_FMT_U32N                     (9)
#define LWE397_IDX_ATTRIBUTE_20_ATTR_FMT_S32                      (10)
#define LWE397_IDX_ATTRIBUTE_20_ATTR_FMT_S32N                     (11)
#define LWE397_IDX_ATTRIBUTE_20_ATTR_FMT_X32                      (12)
#define LWE397_IDX_ATTRIBUTE_20_ATTR_FMT_F32                      (13)
#define LWE397_IDX_ATTRIBUTE_20_ATTR_FMT_H16                      (14)


// Register LWE397_IDX_ATTRIBUTE_21  
#define LWE397_IDX_ATTRIBUTE_21                   (0x115)
#define LWE397_IDX_ATTRIBUTE_21_ATTR_BASE                   31:0

#define LWE397_IDX_ATTRIBUTE_21_SNX_ZERO_PRESERVE                   20:20
#define LWE397_IDX_ATTRIBUTE_21_SNX_ZERO_PRESERVE_DISABLE                 (0)
#define LWE397_IDX_ATTRIBUTE_21_SNX_ZERO_PRESERVE_ENABLE                  (1)

#define LWE397_IDX_ATTRIBUTE_21_ATTR_STRIDE                 19:8

#define LWE397_IDX_ATTRIBUTE_21_ATTR_SIZE                   6:4

#define LWE397_IDX_ATTRIBUTE_21_ATTR_FMT                    3:0
#define LWE397_IDX_ATTRIBUTE_21_ATTR_FMT_U8                       (0)
#define LWE397_IDX_ATTRIBUTE_21_ATTR_FMT_U8N                      (1)
#define LWE397_IDX_ATTRIBUTE_21_ATTR_FMT_S8                       (2)
#define LWE397_IDX_ATTRIBUTE_21_ATTR_FMT_S8N                      (3)
#define LWE397_IDX_ATTRIBUTE_21_ATTR_FMT_U16                      (4)
#define LWE397_IDX_ATTRIBUTE_21_ATTR_FMT_U16N                     (5)
#define LWE397_IDX_ATTRIBUTE_21_ATTR_FMT_S16                      (6)
#define LWE397_IDX_ATTRIBUTE_21_ATTR_FMT_S16N                     (7)
#define LWE397_IDX_ATTRIBUTE_21_ATTR_FMT_U32                      (8)
#define LWE397_IDX_ATTRIBUTE_21_ATTR_FMT_U32N                     (9)
#define LWE397_IDX_ATTRIBUTE_21_ATTR_FMT_S32                      (10)
#define LWE397_IDX_ATTRIBUTE_21_ATTR_FMT_S32N                     (11)
#define LWE397_IDX_ATTRIBUTE_21_ATTR_FMT_X32                      (12)
#define LWE397_IDX_ATTRIBUTE_21_ATTR_FMT_F32                      (13)
#define LWE397_IDX_ATTRIBUTE_21_ATTR_FMT_H16                      (14)


// Register LWE397_IDX_ATTRIBUTE_22  
#define LWE397_IDX_ATTRIBUTE_22                   (0x116)
#define LWE397_IDX_ATTRIBUTE_22_ATTR_BASE                   31:0

#define LWE397_IDX_ATTRIBUTE_22_SNX_ZERO_PRESERVE                   20:20
#define LWE397_IDX_ATTRIBUTE_22_SNX_ZERO_PRESERVE_DISABLE                 (0)
#define LWE397_IDX_ATTRIBUTE_22_SNX_ZERO_PRESERVE_ENABLE                  (1)

#define LWE397_IDX_ATTRIBUTE_22_ATTR_STRIDE                 19:8

#define LWE397_IDX_ATTRIBUTE_22_ATTR_SIZE                   6:4

#define LWE397_IDX_ATTRIBUTE_22_ATTR_FMT                    3:0
#define LWE397_IDX_ATTRIBUTE_22_ATTR_FMT_U8                       (0)
#define LWE397_IDX_ATTRIBUTE_22_ATTR_FMT_U8N                      (1)
#define LWE397_IDX_ATTRIBUTE_22_ATTR_FMT_S8                       (2)
#define LWE397_IDX_ATTRIBUTE_22_ATTR_FMT_S8N                      (3)
#define LWE397_IDX_ATTRIBUTE_22_ATTR_FMT_U16                      (4)
#define LWE397_IDX_ATTRIBUTE_22_ATTR_FMT_U16N                     (5)
#define LWE397_IDX_ATTRIBUTE_22_ATTR_FMT_S16                      (6)
#define LWE397_IDX_ATTRIBUTE_22_ATTR_FMT_S16N                     (7)
#define LWE397_IDX_ATTRIBUTE_22_ATTR_FMT_U32                      (8)
#define LWE397_IDX_ATTRIBUTE_22_ATTR_FMT_U32N                     (9)
#define LWE397_IDX_ATTRIBUTE_22_ATTR_FMT_S32                      (10)
#define LWE397_IDX_ATTRIBUTE_22_ATTR_FMT_S32N                     (11)
#define LWE397_IDX_ATTRIBUTE_22_ATTR_FMT_X32                      (12)
#define LWE397_IDX_ATTRIBUTE_22_ATTR_FMT_F32                      (13)
#define LWE397_IDX_ATTRIBUTE_22_ATTR_FMT_H16                      (14)


// Register LWE397_IDX_ATTRIBUTE_23  
#define LWE397_IDX_ATTRIBUTE_23                   (0x117)
#define LWE397_IDX_ATTRIBUTE_23_ATTR_BASE                   31:0

#define LWE397_IDX_ATTRIBUTE_23_SNX_ZERO_PRESERVE                   20:20
#define LWE397_IDX_ATTRIBUTE_23_SNX_ZERO_PRESERVE_DISABLE                 (0)
#define LWE397_IDX_ATTRIBUTE_23_SNX_ZERO_PRESERVE_ENABLE                  (1)

#define LWE397_IDX_ATTRIBUTE_23_ATTR_STRIDE                 19:8

#define LWE397_IDX_ATTRIBUTE_23_ATTR_SIZE                   6:4

#define LWE397_IDX_ATTRIBUTE_23_ATTR_FMT                    3:0
#define LWE397_IDX_ATTRIBUTE_23_ATTR_FMT_U8                       (0)
#define LWE397_IDX_ATTRIBUTE_23_ATTR_FMT_U8N                      (1)
#define LWE397_IDX_ATTRIBUTE_23_ATTR_FMT_S8                       (2)
#define LWE397_IDX_ATTRIBUTE_23_ATTR_FMT_S8N                      (3)
#define LWE397_IDX_ATTRIBUTE_23_ATTR_FMT_U16                      (4)
#define LWE397_IDX_ATTRIBUTE_23_ATTR_FMT_U16N                     (5)
#define LWE397_IDX_ATTRIBUTE_23_ATTR_FMT_S16                      (6)
#define LWE397_IDX_ATTRIBUTE_23_ATTR_FMT_S16N                     (7)
#define LWE397_IDX_ATTRIBUTE_23_ATTR_FMT_U32                      (8)
#define LWE397_IDX_ATTRIBUTE_23_ATTR_FMT_U32N                     (9)
#define LWE397_IDX_ATTRIBUTE_23_ATTR_FMT_S32                      (10)
#define LWE397_IDX_ATTRIBUTE_23_ATTR_FMT_S32N                     (11)
#define LWE397_IDX_ATTRIBUTE_23_ATTR_FMT_X32                      (12)
#define LWE397_IDX_ATTRIBUTE_23_ATTR_FMT_F32                      (13)
#define LWE397_IDX_ATTRIBUTE_23_ATTR_FMT_H16                      (14)


// Register LWE397_IDX_ATTRIBUTE_24  
#define LWE397_IDX_ATTRIBUTE_24                   (0x118)
#define LWE397_IDX_ATTRIBUTE_24_ATTR_BASE                   31:0

#define LWE397_IDX_ATTRIBUTE_24_SNX_ZERO_PRESERVE                   20:20
#define LWE397_IDX_ATTRIBUTE_24_SNX_ZERO_PRESERVE_DISABLE                 (0)
#define LWE397_IDX_ATTRIBUTE_24_SNX_ZERO_PRESERVE_ENABLE                  (1)

#define LWE397_IDX_ATTRIBUTE_24_ATTR_STRIDE                 19:8

#define LWE397_IDX_ATTRIBUTE_24_ATTR_SIZE                   6:4

#define LWE397_IDX_ATTRIBUTE_24_ATTR_FMT                    3:0
#define LWE397_IDX_ATTRIBUTE_24_ATTR_FMT_U8                       (0)
#define LWE397_IDX_ATTRIBUTE_24_ATTR_FMT_U8N                      (1)
#define LWE397_IDX_ATTRIBUTE_24_ATTR_FMT_S8                       (2)
#define LWE397_IDX_ATTRIBUTE_24_ATTR_FMT_S8N                      (3)
#define LWE397_IDX_ATTRIBUTE_24_ATTR_FMT_U16                      (4)
#define LWE397_IDX_ATTRIBUTE_24_ATTR_FMT_U16N                     (5)
#define LWE397_IDX_ATTRIBUTE_24_ATTR_FMT_S16                      (6)
#define LWE397_IDX_ATTRIBUTE_24_ATTR_FMT_S16N                     (7)
#define LWE397_IDX_ATTRIBUTE_24_ATTR_FMT_U32                      (8)
#define LWE397_IDX_ATTRIBUTE_24_ATTR_FMT_U32N                     (9)
#define LWE397_IDX_ATTRIBUTE_24_ATTR_FMT_S32                      (10)
#define LWE397_IDX_ATTRIBUTE_24_ATTR_FMT_S32N                     (11)
#define LWE397_IDX_ATTRIBUTE_24_ATTR_FMT_X32                      (12)
#define LWE397_IDX_ATTRIBUTE_24_ATTR_FMT_F32                      (13)
#define LWE397_IDX_ATTRIBUTE_24_ATTR_FMT_H16                      (14)


// Register LWE397_IDX_ATTRIBUTE_25  
#define LWE397_IDX_ATTRIBUTE_25                   (0x119)
#define LWE397_IDX_ATTRIBUTE_25_ATTR_BASE                   31:0

#define LWE397_IDX_ATTRIBUTE_25_SNX_ZERO_PRESERVE                   20:20
#define LWE397_IDX_ATTRIBUTE_25_SNX_ZERO_PRESERVE_DISABLE                 (0)
#define LWE397_IDX_ATTRIBUTE_25_SNX_ZERO_PRESERVE_ENABLE                  (1)

#define LWE397_IDX_ATTRIBUTE_25_ATTR_STRIDE                 19:8

#define LWE397_IDX_ATTRIBUTE_25_ATTR_SIZE                   6:4

#define LWE397_IDX_ATTRIBUTE_25_ATTR_FMT                    3:0
#define LWE397_IDX_ATTRIBUTE_25_ATTR_FMT_U8                       (0)
#define LWE397_IDX_ATTRIBUTE_25_ATTR_FMT_U8N                      (1)
#define LWE397_IDX_ATTRIBUTE_25_ATTR_FMT_S8                       (2)
#define LWE397_IDX_ATTRIBUTE_25_ATTR_FMT_S8N                      (3)
#define LWE397_IDX_ATTRIBUTE_25_ATTR_FMT_U16                      (4)
#define LWE397_IDX_ATTRIBUTE_25_ATTR_FMT_U16N                     (5)
#define LWE397_IDX_ATTRIBUTE_25_ATTR_FMT_S16                      (6)
#define LWE397_IDX_ATTRIBUTE_25_ATTR_FMT_S16N                     (7)
#define LWE397_IDX_ATTRIBUTE_25_ATTR_FMT_U32                      (8)
#define LWE397_IDX_ATTRIBUTE_25_ATTR_FMT_U32N                     (9)
#define LWE397_IDX_ATTRIBUTE_25_ATTR_FMT_S32                      (10)
#define LWE397_IDX_ATTRIBUTE_25_ATTR_FMT_S32N                     (11)
#define LWE397_IDX_ATTRIBUTE_25_ATTR_FMT_X32                      (12)
#define LWE397_IDX_ATTRIBUTE_25_ATTR_FMT_F32                      (13)
#define LWE397_IDX_ATTRIBUTE_25_ATTR_FMT_H16                      (14)


// Register LWE397_IDX_ATTRIBUTE_26  
#define LWE397_IDX_ATTRIBUTE_26                   (0x11a)
#define LWE397_IDX_ATTRIBUTE_26_ATTR_BASE                   31:0

#define LWE397_IDX_ATTRIBUTE_26_SNX_ZERO_PRESERVE                   20:20
#define LWE397_IDX_ATTRIBUTE_26_SNX_ZERO_PRESERVE_DISABLE                 (0)
#define LWE397_IDX_ATTRIBUTE_26_SNX_ZERO_PRESERVE_ENABLE                  (1)

#define LWE397_IDX_ATTRIBUTE_26_ATTR_STRIDE                 19:8

#define LWE397_IDX_ATTRIBUTE_26_ATTR_SIZE                   6:4

#define LWE397_IDX_ATTRIBUTE_26_ATTR_FMT                    3:0
#define LWE397_IDX_ATTRIBUTE_26_ATTR_FMT_U8                       (0)
#define LWE397_IDX_ATTRIBUTE_26_ATTR_FMT_U8N                      (1)
#define LWE397_IDX_ATTRIBUTE_26_ATTR_FMT_S8                       (2)
#define LWE397_IDX_ATTRIBUTE_26_ATTR_FMT_S8N                      (3)
#define LWE397_IDX_ATTRIBUTE_26_ATTR_FMT_U16                      (4)
#define LWE397_IDX_ATTRIBUTE_26_ATTR_FMT_U16N                     (5)
#define LWE397_IDX_ATTRIBUTE_26_ATTR_FMT_S16                      (6)
#define LWE397_IDX_ATTRIBUTE_26_ATTR_FMT_S16N                     (7)
#define LWE397_IDX_ATTRIBUTE_26_ATTR_FMT_U32                      (8)
#define LWE397_IDX_ATTRIBUTE_26_ATTR_FMT_U32N                     (9)
#define LWE397_IDX_ATTRIBUTE_26_ATTR_FMT_S32                      (10)
#define LWE397_IDX_ATTRIBUTE_26_ATTR_FMT_S32N                     (11)
#define LWE397_IDX_ATTRIBUTE_26_ATTR_FMT_X32                      (12)
#define LWE397_IDX_ATTRIBUTE_26_ATTR_FMT_F32                      (13)
#define LWE397_IDX_ATTRIBUTE_26_ATTR_FMT_H16                      (14)


// Register LWE397_IDX_ATTRIBUTE_27  
#define LWE397_IDX_ATTRIBUTE_27                   (0x11b)
#define LWE397_IDX_ATTRIBUTE_27_ATTR_BASE                   31:0

#define LWE397_IDX_ATTRIBUTE_27_SNX_ZERO_PRESERVE                   20:20
#define LWE397_IDX_ATTRIBUTE_27_SNX_ZERO_PRESERVE_DISABLE                 (0)
#define LWE397_IDX_ATTRIBUTE_27_SNX_ZERO_PRESERVE_ENABLE                  (1)

#define LWE397_IDX_ATTRIBUTE_27_ATTR_STRIDE                 19:8

#define LWE397_IDX_ATTRIBUTE_27_ATTR_SIZE                   6:4

#define LWE397_IDX_ATTRIBUTE_27_ATTR_FMT                    3:0
#define LWE397_IDX_ATTRIBUTE_27_ATTR_FMT_U8                       (0)
#define LWE397_IDX_ATTRIBUTE_27_ATTR_FMT_U8N                      (1)
#define LWE397_IDX_ATTRIBUTE_27_ATTR_FMT_S8                       (2)
#define LWE397_IDX_ATTRIBUTE_27_ATTR_FMT_S8N                      (3)
#define LWE397_IDX_ATTRIBUTE_27_ATTR_FMT_U16                      (4)
#define LWE397_IDX_ATTRIBUTE_27_ATTR_FMT_U16N                     (5)
#define LWE397_IDX_ATTRIBUTE_27_ATTR_FMT_S16                      (6)
#define LWE397_IDX_ATTRIBUTE_27_ATTR_FMT_S16N                     (7)
#define LWE397_IDX_ATTRIBUTE_27_ATTR_FMT_U32                      (8)
#define LWE397_IDX_ATTRIBUTE_27_ATTR_FMT_U32N                     (9)
#define LWE397_IDX_ATTRIBUTE_27_ATTR_FMT_S32                      (10)
#define LWE397_IDX_ATTRIBUTE_27_ATTR_FMT_S32N                     (11)
#define LWE397_IDX_ATTRIBUTE_27_ATTR_FMT_X32                      (12)
#define LWE397_IDX_ATTRIBUTE_27_ATTR_FMT_F32                      (13)
#define LWE397_IDX_ATTRIBUTE_27_ATTR_FMT_H16                      (14)


// Register LWE397_IDX_ATTRIBUTE_28  
#define LWE397_IDX_ATTRIBUTE_28                   (0x11c)
#define LWE397_IDX_ATTRIBUTE_28_ATTR_BASE                   31:0

#define LWE397_IDX_ATTRIBUTE_28_SNX_ZERO_PRESERVE                   20:20
#define LWE397_IDX_ATTRIBUTE_28_SNX_ZERO_PRESERVE_DISABLE                 (0)
#define LWE397_IDX_ATTRIBUTE_28_SNX_ZERO_PRESERVE_ENABLE                  (1)

#define LWE397_IDX_ATTRIBUTE_28_ATTR_STRIDE                 19:8

#define LWE397_IDX_ATTRIBUTE_28_ATTR_SIZE                   6:4

#define LWE397_IDX_ATTRIBUTE_28_ATTR_FMT                    3:0
#define LWE397_IDX_ATTRIBUTE_28_ATTR_FMT_U8                       (0)
#define LWE397_IDX_ATTRIBUTE_28_ATTR_FMT_U8N                      (1)
#define LWE397_IDX_ATTRIBUTE_28_ATTR_FMT_S8                       (2)
#define LWE397_IDX_ATTRIBUTE_28_ATTR_FMT_S8N                      (3)
#define LWE397_IDX_ATTRIBUTE_28_ATTR_FMT_U16                      (4)
#define LWE397_IDX_ATTRIBUTE_28_ATTR_FMT_U16N                     (5)
#define LWE397_IDX_ATTRIBUTE_28_ATTR_FMT_S16                      (6)
#define LWE397_IDX_ATTRIBUTE_28_ATTR_FMT_S16N                     (7)
#define LWE397_IDX_ATTRIBUTE_28_ATTR_FMT_U32                      (8)
#define LWE397_IDX_ATTRIBUTE_28_ATTR_FMT_U32N                     (9)
#define LWE397_IDX_ATTRIBUTE_28_ATTR_FMT_S32                      (10)
#define LWE397_IDX_ATTRIBUTE_28_ATTR_FMT_S32N                     (11)
#define LWE397_IDX_ATTRIBUTE_28_ATTR_FMT_X32                      (12)
#define LWE397_IDX_ATTRIBUTE_28_ATTR_FMT_F32                      (13)
#define LWE397_IDX_ATTRIBUTE_28_ATTR_FMT_H16                      (14)


// Register LWE397_IDX_ATTRIBUTE_29  
#define LWE397_IDX_ATTRIBUTE_29                   (0x11d)
#define LWE397_IDX_ATTRIBUTE_29_ATTR_BASE                   31:0

#define LWE397_IDX_ATTRIBUTE_29_SNX_ZERO_PRESERVE                   20:20
#define LWE397_IDX_ATTRIBUTE_29_SNX_ZERO_PRESERVE_DISABLE                 (0)
#define LWE397_IDX_ATTRIBUTE_29_SNX_ZERO_PRESERVE_ENABLE                  (1)

#define LWE397_IDX_ATTRIBUTE_29_ATTR_STRIDE                 19:8

#define LWE397_IDX_ATTRIBUTE_29_ATTR_SIZE                   6:4

#define LWE397_IDX_ATTRIBUTE_29_ATTR_FMT                    3:0
#define LWE397_IDX_ATTRIBUTE_29_ATTR_FMT_U8                       (0)
#define LWE397_IDX_ATTRIBUTE_29_ATTR_FMT_U8N                      (1)
#define LWE397_IDX_ATTRIBUTE_29_ATTR_FMT_S8                       (2)
#define LWE397_IDX_ATTRIBUTE_29_ATTR_FMT_S8N                      (3)
#define LWE397_IDX_ATTRIBUTE_29_ATTR_FMT_U16                      (4)
#define LWE397_IDX_ATTRIBUTE_29_ATTR_FMT_U16N                     (5)
#define LWE397_IDX_ATTRIBUTE_29_ATTR_FMT_S16                      (6)
#define LWE397_IDX_ATTRIBUTE_29_ATTR_FMT_S16N                     (7)
#define LWE397_IDX_ATTRIBUTE_29_ATTR_FMT_U32                      (8)
#define LWE397_IDX_ATTRIBUTE_29_ATTR_FMT_U32N                     (9)
#define LWE397_IDX_ATTRIBUTE_29_ATTR_FMT_S32                      (10)
#define LWE397_IDX_ATTRIBUTE_29_ATTR_FMT_S32N                     (11)
#define LWE397_IDX_ATTRIBUTE_29_ATTR_FMT_X32                      (12)
#define LWE397_IDX_ATTRIBUTE_29_ATTR_FMT_F32                      (13)
#define LWE397_IDX_ATTRIBUTE_29_ATTR_FMT_H16                      (14)


// Register LWE397_IDX_ATTRIBUTE_30  
#define LWE397_IDX_ATTRIBUTE_30                   (0x11e)
#define LWE397_IDX_ATTRIBUTE_30_ATTR_BASE                   31:0

#define LWE397_IDX_ATTRIBUTE_30_SNX_ZERO_PRESERVE                   20:20
#define LWE397_IDX_ATTRIBUTE_30_SNX_ZERO_PRESERVE_DISABLE                 (0)
#define LWE397_IDX_ATTRIBUTE_30_SNX_ZERO_PRESERVE_ENABLE                  (1)

#define LWE397_IDX_ATTRIBUTE_30_ATTR_STRIDE                 19:8

#define LWE397_IDX_ATTRIBUTE_30_ATTR_SIZE                   6:4

#define LWE397_IDX_ATTRIBUTE_30_ATTR_FMT                    3:0
#define LWE397_IDX_ATTRIBUTE_30_ATTR_FMT_U8                       (0)
#define LWE397_IDX_ATTRIBUTE_30_ATTR_FMT_U8N                      (1)
#define LWE397_IDX_ATTRIBUTE_30_ATTR_FMT_S8                       (2)
#define LWE397_IDX_ATTRIBUTE_30_ATTR_FMT_S8N                      (3)
#define LWE397_IDX_ATTRIBUTE_30_ATTR_FMT_U16                      (4)
#define LWE397_IDX_ATTRIBUTE_30_ATTR_FMT_U16N                     (5)
#define LWE397_IDX_ATTRIBUTE_30_ATTR_FMT_S16                      (6)
#define LWE397_IDX_ATTRIBUTE_30_ATTR_FMT_S16N                     (7)
#define LWE397_IDX_ATTRIBUTE_30_ATTR_FMT_U32                      (8)
#define LWE397_IDX_ATTRIBUTE_30_ATTR_FMT_U32N                     (9)
#define LWE397_IDX_ATTRIBUTE_30_ATTR_FMT_S32                      (10)
#define LWE397_IDX_ATTRIBUTE_30_ATTR_FMT_S32N                     (11)
#define LWE397_IDX_ATTRIBUTE_30_ATTR_FMT_X32                      (12)
#define LWE397_IDX_ATTRIBUTE_30_ATTR_FMT_F32                      (13)
#define LWE397_IDX_ATTRIBUTE_30_ATTR_FMT_H16                      (14)


// Register LWE397_IDX_ATTRIBUTE_31  
#define LWE397_IDX_ATTRIBUTE_31                   (0x11f)
#define LWE397_IDX_ATTRIBUTE_31_ATTR_BASE                   31:0

#define LWE397_IDX_ATTRIBUTE_31_SNX_ZERO_PRESERVE                   20:20
#define LWE397_IDX_ATTRIBUTE_31_SNX_ZERO_PRESERVE_DISABLE                 (0)
#define LWE397_IDX_ATTRIBUTE_31_SNX_ZERO_PRESERVE_ENABLE                  (1)

#define LWE397_IDX_ATTRIBUTE_31_ATTR_STRIDE                 19:8

#define LWE397_IDX_ATTRIBUTE_31_ATTR_SIZE                   6:4

#define LWE397_IDX_ATTRIBUTE_31_ATTR_FMT                    3:0
#define LWE397_IDX_ATTRIBUTE_31_ATTR_FMT_U8                       (0)
#define LWE397_IDX_ATTRIBUTE_31_ATTR_FMT_U8N                      (1)
#define LWE397_IDX_ATTRIBUTE_31_ATTR_FMT_S8                       (2)
#define LWE397_IDX_ATTRIBUTE_31_ATTR_FMT_S8N                      (3)
#define LWE397_IDX_ATTRIBUTE_31_ATTR_FMT_U16                      (4)
#define LWE397_IDX_ATTRIBUTE_31_ATTR_FMT_U16N                     (5)
#define LWE397_IDX_ATTRIBUTE_31_ATTR_FMT_S16                      (6)
#define LWE397_IDX_ATTRIBUTE_31_ATTR_FMT_S16N                     (7)
#define LWE397_IDX_ATTRIBUTE_31_ATTR_FMT_U32                      (8)
#define LWE397_IDX_ATTRIBUTE_31_ATTR_FMT_U32N                     (9)
#define LWE397_IDX_ATTRIBUTE_31_ATTR_FMT_S32                      (10)
#define LWE397_IDX_ATTRIBUTE_31_ATTR_FMT_S32N                     (11)
#define LWE397_IDX_ATTRIBUTE_31_ATTR_FMT_X32                      (12)
#define LWE397_IDX_ATTRIBUTE_31_ATTR_FMT_F32                      (13)
#define LWE397_IDX_ATTRIBUTE_31_ATTR_FMT_H16                      (14)


// Register LWE397_IDX_ATTR_MASK_0  
#define LWE397_IDX_ATTR_MASK_0                    (0x120)
#define LWE397_IDX_ATTR_MASK_0_INPUT_ATTR_MASK                      31:16

#define LWE397_IDX_ATTR_MASK_0_OUTPUT_ATTR_MASK                     15:0


// Register LWE397_IDX_INDEX_BASE_0  
#define LWE397_IDX_INDEX_BASE_0                   (0x121)
#define LWE397_IDX_INDEX_BASE_0_INDEX_BASE                  31:0


// Register LWE397_IDX_SET_PRIM_0  
#define LWE397_IDX_SET_PRIM_0                     (0x122)
#define LWE397_IDX_SET_PRIM_0_ILWALIDATE_VTXCACHE                   31:31

#define LWE397_IDX_SET_PRIM_0_ILWALIDATE_DMACACHE                   30:30

#define LWE397_IDX_SET_PRIM_0_DRAW_MODE                     29:28
#define LWE397_IDX_SET_PRIM_0_DRAW_MODE_ARRAY                     (0)
#define LWE397_IDX_SET_PRIM_0_DRAW_MODE_ELE8                      (1)
#define LWE397_IDX_SET_PRIM_0_DRAW_MODE_ELE16                     (2)
#define LWE397_IDX_SET_PRIM_0_DRAW_MODE_ELE32                     (3)

#define LWE397_IDX_SET_PRIM_0_FLAT_VTX                      27:27
#define LWE397_IDX_SET_PRIM_0_FLAT_VTX_FIRST                      (0)
#define LWE397_IDX_SET_PRIM_0_FLAT_VTX_LAST                       (1)

#define LWE397_IDX_SET_PRIM_0_PRIM_TYPE                     26:24
#define LWE397_IDX_SET_PRIM_0_PRIM_TYPE_POINTS                    (0)
#define LWE397_IDX_SET_PRIM_0_PRIM_TYPE_LINES                     (1)
#define LWE397_IDX_SET_PRIM_0_PRIM_TYPE_LINE_LOOP                 (2)
#define LWE397_IDX_SET_PRIM_0_PRIM_TYPE_LINE_STRIP                        (3)
#define LWE397_IDX_SET_PRIM_0_PRIM_TYPE_TRIS                      (4)
#define LWE397_IDX_SET_PRIM_0_PRIM_TYPE_TRI_STRIP                 (5)
#define LWE397_IDX_SET_PRIM_0_PRIM_TYPE_TRI_FAN                   (6)

#define LWE397_IDX_SET_PRIM_0_PIVOT_VTX                     19:0


// Register LWE397_IDX_DRAW_PRIM_0  
#define LWE397_IDX_DRAW_PRIM_0                    (0x123)
#define LWE397_IDX_DRAW_PRIM_0_VTX_COUNT                    31:20

#define LWE397_IDX_DRAW_PRIM_0_START_VTX                    19:0


// Register LWE397_IDX_IDX_CTL_0  
#define LWE397_IDX_IDX_CTL_0                      (0x124)
#define LWE397_IDX_IDX_CTL_0_INDEX_TOO_LARGE                        5:5

#define LWE397_IDX_IDX_CTL_0_FORCE_TRANSFORM                        4:4

#define LWE397_IDX_IDX_CTL_0_DMACACHE_DISABLE                       3:3

#define LWE397_IDX_IDX_CTL_0_LATE_BINDING                   2:2

#define LWE397_IDX_IDX_CTL_0_VAR_OBUF_SIZE                  1:1

#define LWE397_IDX_IDX_CTL_0_VAR_IBUF_SIZE                  0:0


// Register LWE397_IDX_IDX_STAT_0  
#define LWE397_IDX_IDX_STAT_0                     (0x125)
#define LWE397_IDX_IDX_STAT_0_IBUF_INUSE                    7:0


// Register LWE397_IDX_LW_MCCIF_FIFOCTRL_RO_0  
#define LWE397_IDX_LW_MCCIF_FIFOCTRL_RO_0                 (0x126)
#define LWE397_IDX_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_WRCL_MCLE2X                      0:0
#define LWE397_IDX_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_WRCL_MCLE2X_DISABLE                    (0)
#define LWE397_IDX_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_WRCL_MCLE2X_ENABLE                     (1)

#define LWE397_IDX_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_RDMC_RDFAST                      1:1
#define LWE397_IDX_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_RDMC_RDFAST_DISABLE                    (0)
#define LWE397_IDX_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_RDMC_RDFAST_ENABLE                     (1)

#define LWE397_IDX_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_WRMC_CLLE2X                      2:2
#define LWE397_IDX_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_WRMC_CLLE2X_DISABLE                    (0)
#define LWE397_IDX_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_WRMC_CLLE2X_ENABLE                     (1)

#define LWE397_IDX_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_RDCL_RDFAST                      3:3
#define LWE397_IDX_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_RDCL_RDFAST_DISABLE                    (0)
#define LWE397_IDX_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_RDCL_RDFAST_ENABLE                     (1)

#define LWE397_IDX_LW_MCCIF_FIFOCTRL_RO_0_LW_WCLK_OVERRIDE                  16:16

#define LWE397_IDX_LW_MCCIF_FIFOCTRL_RO_0_LW_RCLK_OVERRIDE                  17:17

#define NUM_IBUF        8
#define NUM_OBUF        16
#define IBUF_RAM_SIZE   64
#define OBUF_RAM_SIZE   64

// Register LWE397_VPE_MODE_0  
#define LWE397_VPE_MODE_0                 (0x200)
#define LWE397_VPE_MODE_0_ZERO_MODE                 4:4
#define LWE397_VPE_MODE_0_ZERO_MODE_UNEQUAL                       (0)
#define LWE397_VPE_MODE_0_ZERO_MODE_EQUAL                 (1)

#define LWE397_VPE_MODE_0_SHADER_VERSION                    1:0
#define LWE397_VPE_MODE_0_SHADER_VERSION_V1                       (2)
#define LWE397_VPE_MODE_0_SHADER_VERSION_V2                       (3)
#define LWE397_VPE_MODE_0_SHADER_VERSION_V3                       (1)


// Register LWE397_VPE_TIMEOUT_0  
#define LWE397_VPE_TIMEOUT_0                      (0x201)
#define LWE397_VPE_TIMEOUT_0_MAX_INSN_COUNT                 15:0


// Register LWE397_VPE_CONST_READ_LIMIT_0  
#define LWE397_VPE_CONST_READ_LIMIT_0                     (0x202)
#define LWE397_VPE_CONST_READ_LIMIT_0_MAX_CONST_INDEX                       31:16

#define LWE397_VPE_CONST_READ_LIMIT_0_MIN_CONST_INDEX                       15:0


// Register LWE397_VPE_BRANCHBITS_0  
#define LWE397_VPE_BRANCHBITS_0                   (0x203)
#define LWE397_VPE_BRANCHBITS_0_BRANCHBITS                  31:0


// Register LWE397_VPE_START_0  
#define LWE397_VPE_START_0                        (0x204)
#define LWE397_VPE_START_0_START_ADDR                       7:0


// Register LWE397_VPE_INST_OFFSET_0  
#define LWE397_VPE_INST_OFFSET_0                  (0x205)
#define LWE397_VPE_INST_OFFSET_0_INDEX                      9:0


// Register LWE397_VPE_INST_DATA_0  
#define LWE397_VPE_INST_DATA_0                    (0x206)
#define LWE397_VPE_INST_DATA_0_INST_DATA                    31:0


// Register LWE397_VPE_CONST_OFFSET_0  
#define LWE397_VPE_CONST_OFFSET_0                 (0x207)
#define LWE397_VPE_CONST_OFFSET_0_INDEX                     9:0


// Register LWE397_VPE_CONST_DATA_0  
#define LWE397_VPE_CONST_DATA_0                   (0x208)
#define LWE397_VPE_CONST_DATA_0_CONST_DATA                  31:0


// Register LWE397_VPE_GEOM_STALL_0  
#define LWE397_VPE_GEOM_STALL_0                   (0x209)
#define LWE397_VPE_GEOM_STALL_0_VPE_STALL                   7:0

#define LWE397_VPE_GEOM_STALL_0_VPE_FLUSH                   16:16


// Register LWE397_VPE_VPE_CTRL_0  
#define LWE397_VPE_VPE_CTRL_0                     (0x20a)
#define LWE397_VPE_VPE_CTRL_0_VPE_CTRL_VPEOR_CLKEN_OVR                      11:11

#define LWE397_VPE_VPE_CTRL_0_VPE_CTRL_VPERF_CLKEN_OVR                      10:10

#define LWE397_VPE_VPE_CTRL_0_VPE_CTRL_VPEOD_CLKEN_OVR                      9:9

#define LWE397_VPE_VPE_CTRL_0_VPE_CTRL_VPEOB_CLKEN_OVR                      8:8

#define LWE397_VPE_VPE_CTRL_0_VPE_CTRL_VPEIB_CLKEN_OVR                      7:7

#define LWE397_VPE_VPE_CTRL_0_VPE_CTRL_VPECR_CLKEN_OVR                      6:6

#define LWE397_VPE_VPE_CTRL_0_VPE_CTRL_VPEIR_CLKEN_OVR                      5:5

#define LWE397_VPE_VPE_CTRL_0_VPE_CTRL_VPEDP_CLKEN_OVR                      4:4

#define LWE397_VPE_VPE_CTRL_0_VPE_CTRL_SPARE0                       3:3

#define LWE397_VPE_VPE_CTRL_0_VPE_CTRL_THR2_DISABLE                 2:2

#define LWE397_VPE_VPE_CTRL_0_VPE_CTRL_THR1_DISABLE                 1:1

#define LWE397_VPE_VPE_CTRL_0_VPE_CTRL_THR0_DISABLE                 0:0


// Register LWE397_VPE_VPE_DEBUG_0  
#define LWE397_VPE_VPE_DEBUG_0                    (0x20b)
#define LWE397_VPE_VPE_DEBUG_0_VPE_DEBUG_SPARE                      7:7

#define LWE397_VPE_VPE_DEBUG_0_VPE_DEBUG_VTF_SERIAL                 6:6

#define LWE397_VPE_VPE_DEBUG_0_VPE_DEBUG_THREAD_EXCLUSIVE                   5:5

#define LWE397_VPE_VPE_DEBUG_0_VPE_DEBUG_SERIAL_THREADS                     4:4

#define LWE397_VPE_VPE_DEBUG_0_VPE_DEBUG_OLDEST_FIRST                       3:3

#define LWE397_VPE_VPE_DEBUG_0_VPE_DEBUG_SERIALIZE_MODES                    2:2

#define LWE397_VPE_VPE_DEBUG_0_VPE_DEBUG_OBUF_SIZE                  1:1

#define LWE397_VPE_VPE_DEBUG_0_VPE_DEBUG_IBUF_SIZE                  0:0

/*
#define VPE_INSTR_REG_NOP       63
#define VPE_INSTR_OUT_NOP       31

// Packet VPE_4X_INSTR
#define VPE_4X_INSTR_SIZE 127

#define VPE_4X_INSTR_LAST_ROW                   0

#define VPE_4X_INSTR_CTX_INDX_ROW                       0

#define VPE_4X_INSTR_OUT_ROW                    0

#define VPE_4X_INSTR_SRT_ADDR_ROW                       0

#define VPE_4X_INSTR_VWE_ROW                    0

#define VPE_4X_INSTR_SWE_ROW                    0

#define VPE_4X_INSTR_RC_TYPE_ROW                        0
#define VPE_4X_INSTR_RC_TYPE_ILWAL                      (0)
#define VPE_4X_INSTR_RC_TYPE_REG                        (1)
#define VPE_4X_INSTR_RC_TYPE_BUF                        (2)
#define VPE_4X_INSTR_RC_TYPE_CTX                        (3)

#define VPE_4X_INSTR_RC_ROW                     0

#define VPE_4X_INSTR_RC_W_EXTR_ROW                      0
#define VPE_4X_INSTR_RC_W_EXTR_X                        (0)
#define VPE_4X_INSTR_RC_W_EXTR_Y                        (1)
#define VPE_4X_INSTR_RC_W_EXTR_Z                        (2)
#define VPE_4X_INSTR_RC_W_EXTR_W                        (3)

#define VPE_4X_INSTR_RC_Z_EXTR_ROW                      0
#define VPE_4X_INSTR_RC_Z_EXTR_X                        (0)
#define VPE_4X_INSTR_RC_Z_EXTR_Y                        (1)
#define VPE_4X_INSTR_RC_Z_EXTR_Z                        (2)
#define VPE_4X_INSTR_RC_Z_EXTR_W                        (3)

#define VPE_4X_INSTR_RC_Y_EXTR_ROW                      0
#define VPE_4X_INSTR_RC_Y_EXTR_X                        (0)
#define VPE_4X_INSTR_RC_Y_EXTR_Y                        (1)
#define VPE_4X_INSTR_RC_Y_EXTR_Z                        (2)
#define VPE_4X_INSTR_RC_Y_EXTR_W                        (3)

#define VPE_4X_INSTR_RC_X_EXTR_ROW                      0
#define VPE_4X_INSTR_RC_X_EXTR_X                        (0)
#define VPE_4X_INSTR_RC_X_EXTR_Y                        (1)
#define VPE_4X_INSTR_RC_X_EXTR_Z                        (2)
#define VPE_4X_INSTR_RC_X_EXTR_W                        (3)

#define VPE_4X_INSTR_RC_NEG_ROW                 0

#define VPE_4X_INSTR_RB_TYPE_ROW                        0
#define VPE_4X_INSTR_RB_TYPE_ILWAL                      (0)
#define VPE_4X_INSTR_RB_TYPE_REG                        (1)
#define VPE_4X_INSTR_RB_TYPE_BUF                        (2)
#define VPE_4X_INSTR_RB_TYPE_CTX                        (3)

#define VPE_4X_INSTR_RB_ROW                     0

#define VPE_4X_INSTR_RB_W_EXTR_ROW                      0
#define VPE_4X_INSTR_RB_W_EXTR_X                        (0)
#define VPE_4X_INSTR_RB_W_EXTR_Y                        (1)
#define VPE_4X_INSTR_RB_W_EXTR_Z                        (2)
#define VPE_4X_INSTR_RB_W_EXTR_W                        (3)

#define VPE_4X_INSTR_RB_Z_EXTR_ROW                      0
#define VPE_4X_INSTR_RB_Z_EXTR_X                        (0)
#define VPE_4X_INSTR_RB_Z_EXTR_Y                        (1)
#define VPE_4X_INSTR_RB_Z_EXTR_Z                        (2)
#define VPE_4X_INSTR_RB_Z_EXTR_W                        (3)

#define VPE_4X_INSTR_RB_Y_EXTR_ROW                      0
#define VPE_4X_INSTR_RB_Y_EXTR_X                        (0)
#define VPE_4X_INSTR_RB_Y_EXTR_Y                        (1)
#define VPE_4X_INSTR_RB_Y_EXTR_Z                        (2)
#define VPE_4X_INSTR_RB_Y_EXTR_W                        (3)

#define VPE_4X_INSTR_RB_X_EXTR_ROW                      0
#define VPE_4X_INSTR_RB_X_EXTR_X                        (0)
#define VPE_4X_INSTR_RB_X_EXTR_Y                        (1)
#define VPE_4X_INSTR_RB_X_EXTR_Z                        (2)
#define VPE_4X_INSTR_RB_X_EXTR_W                        (3)

#define VPE_4X_INSTR_RB_NEG_ROW                 0

#define VPE_4X_INSTR_RA_TYPE_ROW                        0
#define VPE_4X_INSTR_RA_TYPE_ILWAL                      (0)
#define VPE_4X_INSTR_RA_TYPE_REG                        (1)
#define VPE_4X_INSTR_RA_TYPE_BUF                        (2)
#define VPE_4X_INSTR_RA_TYPE_CTX                        (3)

#define VPE_4X_INSTR_RA_ROW                     0

#define VPE_4X_INSTR_RA_W_EXTR_ROW                      0
#define VPE_4X_INSTR_RA_W_EXTR_X                        (0)
#define VPE_4X_INSTR_RA_W_EXTR_Y                        (1)
#define VPE_4X_INSTR_RA_W_EXTR_Z                        (2)
#define VPE_4X_INSTR_RA_W_EXTR_W                        (3)

#define VPE_4X_INSTR_RA_Z_EXTR_ROW                      0
#define VPE_4X_INSTR_RA_Z_EXTR_X                        (0)
#define VPE_4X_INSTR_RA_Z_EXTR_Y                        (1)
#define VPE_4X_INSTR_RA_Z_EXTR_Z                        (2)
#define VPE_4X_INSTR_RA_Z_EXTR_W                        (3)

#define VPE_4X_INSTR_RA_Y_EXTR_ROW                      0
#define VPE_4X_INSTR_RA_Y_EXTR_X                        (0)
#define VPE_4X_INSTR_RA_Y_EXTR_Y                        (1)
#define VPE_4X_INSTR_RA_Y_EXTR_Z                        (2)
#define VPE_4X_INSTR_RA_Y_EXTR_W                        (3)

#define VPE_4X_INSTR_RA_X_EXTR_ROW                      0
#define VPE_4X_INSTR_RA_X_EXTR_X                        (0)
#define VPE_4X_INSTR_RA_X_EXTR_Y                        (1)
#define VPE_4X_INSTR_RA_X_EXTR_Z                        (2)
#define VPE_4X_INSTR_RA_X_EXTR_W                        (3)

#define VPE_4X_INSTR_RA_NEG_ROW                 0

#define VPE_4X_INSTR_IBUF_ADDR_ROW                      0

#define VPE_4X_INSTR_CTX_ADDR_ROW                       0

#define VPE_4X_INSTR_OPCODE_V_ROW                       0
#define VPE_4X_INSTR_OPCODE_V_NOP                       (0)
#define VPE_4X_INSTR_OPCODE_V_MOV                       (1)
#define VPE_4X_INSTR_OPCODE_V_MUL                       (2)
#define VPE_4X_INSTR_OPCODE_V_ADD                       (3)
#define VPE_4X_INSTR_OPCODE_V_MAD                       (4)
#define VPE_4X_INSTR_OPCODE_V_DP3                       (5)
#define VPE_4X_INSTR_OPCODE_V_DPH                       (6)
#define VPE_4X_INSTR_OPCODE_V_DP4                       (7)
#define VPE_4X_INSTR_OPCODE_V_DST                       (8)
#define VPE_4X_INSTR_OPCODE_V_MIN                       (9)
#define VPE_4X_INSTR_OPCODE_V_MAX                       (10)
#define VPE_4X_INSTR_OPCODE_V_SLT                       (11)
#define VPE_4X_INSTR_OPCODE_V_SGE                       (12)
#define VPE_4X_INSTR_OPCODE_V_ARL                       (13)
#define VPE_4X_INSTR_OPCODE_V_FRC                       (14)
#define VPE_4X_INSTR_OPCODE_V_FLR                       (15)
#define VPE_4X_INSTR_OPCODE_V_SEQ                       (16)
#define VPE_4X_INSTR_OPCODE_V_SFL                       (17)
#define VPE_4X_INSTR_OPCODE_V_SGT                       (18)
#define VPE_4X_INSTR_OPCODE_V_SLE                       (19)
#define VPE_4X_INSTR_OPCODE_V_SNE                       (20)
#define VPE_4X_INSTR_OPCODE_V_STR                       (21)
#define VPE_4X_INSTR_OPCODE_V_SSG                       (22)
#define VPE_4X_INSTR_OPCODE_V_ARR                       (23)
#define VPE_4X_INSTR_OPCODE_V_MVA                       (24)
#define VPE_4X_INSTR_OPCODE_V_TXL                       (25)
#define VPE_4X_INSTR_OPCODE_V_PSH                       (26)
#define VPE_4X_INSTR_OPCODE_V_POP                       (27)
#define VPE_4X_INSTR_OPCODE_V_RSV0                      (28)
#define VPE_4X_INSTR_OPCODE_V_RSV1                      (29)
#define VPE_4X_INSTR_OPCODE_V_RSV2                      (30)
#define VPE_4X_INSTR_OPCODE_V_RSV3                      (31)

#define VPE_4X_INSTR_OPCODE_S_ROW                       0
#define VPE_4X_INSTR_OPCODE_S_NOP                       (0)
#define VPE_4X_INSTR_OPCODE_S_MOV                       (1)
#define VPE_4X_INSTR_OPCODE_S_RCP                       (2)
#define VPE_4X_INSTR_OPCODE_S_RCC                       (3)
#define VPE_4X_INSTR_OPCODE_S_RSQ                       (4)
#define VPE_4X_INSTR_OPCODE_S_EXP                       (5)
#define VPE_4X_INSTR_OPCODE_S_LOG                       (6)
#define VPE_4X_INSTR_OPCODE_S_LIT                       (7)
#define VPE_4X_INSTR_OPCODE_S_BRA                       (8)
#define VPE_4X_INSTR_OPCODE_S_BRI                       (9)
#define VPE_4X_INSTR_OPCODE_S_CLA                       (10)
#define VPE_4X_INSTR_OPCODE_S_CLI                       (11)
#define VPE_4X_INSTR_OPCODE_S_RET                       (12)
#define VPE_4X_INSTR_OPCODE_S_LG2                       (13)
#define VPE_4X_INSTR_OPCODE_S_EX2                       (14)
#define VPE_4X_INSTR_OPCODE_S_SIN                       (15)
#define VPE_4X_INSTR_OPCODE_S_COS                       (16)
#define VPE_4X_INSTR_OPCODE_S_BRB                       (17)
#define VPE_4X_INSTR_OPCODE_S_CLB                       (18)
#define VPE_4X_INSTR_OPCODE_S_PSH                       (19)
#define VPE_4X_INSTR_OPCODE_S_POP                       (20)
#define VPE_4X_INSTR_OPCODE_S_RSV0                      (21)

#define VPE_4X_INSTR_SCALAR_SEL_ROW                     0

#define VPE_4X_INSTR_RCC_W_EXTR_ROW                     0
#define VPE_4X_INSTR_RCC_W_EXTR_X                       (0)
#define VPE_4X_INSTR_RCC_W_EXTR_Y                       (1)
#define VPE_4X_INSTR_RCC_W_EXTR_Z                       (2)
#define VPE_4X_INSTR_RCC_W_EXTR_W                       (3)

#define VPE_4X_INSTR_RCC_Z_EXTR_ROW                     0
#define VPE_4X_INSTR_RCC_Z_EXTR_X                       (0)
#define VPE_4X_INSTR_RCC_Z_EXTR_Y                       (1)
#define VPE_4X_INSTR_RCC_Z_EXTR_Z                       (2)
#define VPE_4X_INSTR_RCC_Z_EXTR_W                       (3)

#define VPE_4X_INSTR_RCC_Y_EXTR_ROW                     0
#define VPE_4X_INSTR_RCC_Y_EXTR_X                       (0)
#define VPE_4X_INSTR_RCC_Y_EXTR_Y                       (1)
#define VPE_4X_INSTR_RCC_Y_EXTR_Z                       (2)
#define VPE_4X_INSTR_RCC_Y_EXTR_W                       (3)

#define VPE_4X_INSTR_RCC_X_EXTR_ROW                     0
#define VPE_4X_INSTR_RCC_X_EXTR_X                       (0)
#define VPE_4X_INSTR_RCC_X_EXTR_Y                       (1)
#define VPE_4X_INSTR_RCC_X_EXTR_Z                       (2)
#define VPE_4X_INSTR_RCC_X_EXTR_W                       (3)

#define VPE_4X_INSTR_RCC_COMPARE_ROW                    0
#define VPE_4X_INSTR_RCC_COMPARE_FALSE                  (0)
#define VPE_4X_INSTR_RCC_COMPARE_LT                     (1)
#define VPE_4X_INSTR_RCC_COMPARE_EQ                     (2)
#define VPE_4X_INSTR_RCC_COMPARE_LE                     (3)
#define VPE_4X_INSTR_RCC_COMPARE_GT                     (4)
#define VPE_4X_INSTR_RCC_COMPARE_NE                     (5)
#define VPE_4X_INSTR_RCC_COMPARE_GE                     (6)
#define VPE_4X_INSTR_RCC_COMPARE_TRUE                   (7)

#define VPE_4X_INSTR_MOD_WE_ROW                 0

#define VPE_4X_INSTR_RCC_WEN_ROW                        0

#define VPE_4X_INSTR_RT_ADDR_ROW                        0

#define VPE_4X_INSTR_RA_ABS_ROW                 0

#define VPE_4X_INSTR_RB_ABS_ROW                 0

#define VPE_4X_INSTR_RC_ABS_ROW                 0

#define VPE_4X_INSTR_OFFREG_RA_ROW                      0

#define VPE_4X_INSTR_CC_SEL_ROW                 0

#define VPE_4X_INSTR_SATURATE_ROW                       0

#define VPE_4X_INSTR_IBUF_INDX_ROW                      0

#define VPE_4X_INSTR_OBUF_INDX_ROW                      0

#define VPE_4X_INSTR_CC_WR_SEL_ROW                      0
#define VPE_4X_INSTR_CC_WR_SEL_SCALAR                   (0)
#define VPE_4X_INSTR_CC_WR_SEL_VECTOR                   (1)

#define VPE_4X_INSTR_OUT_SEL_ROW                        0
#define VPE_4X_INSTR_OUT_SEL_SCALAR                     (0)
#define VPE_4X_INSTR_OUT_SEL_VECTOR                     (1)
*/

// Register LWE397_SU_INST_0  
#define LWE397_SU_INST_0                  (0x300)
#define LWE397_SU_INST_0_SRC                        1:0
#define LWE397_SU_INST_0_SRC_VPE                  (0)
#define LWE397_SU_INST_0_SRC_Z                    (1)

#define LWE397_SU_INST_0_VC_ROW                     6:3

#define LWE397_SU_INST_0_TRAM_ROW                   14:9

#define LWE397_SU_INST_0_P0_LINE_WIDTH                      16:16
#define LWE397_SU_INST_0_P0_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_0_P0_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_0_P0_LINE_LENGTH                     17:17
#define LWE397_SU_INST_0_P0_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_0_P0_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_0_P0_POINT                   19:18
#define LWE397_SU_INST_0_P0_POINT_DISABLE                 (0)
#define LWE397_SU_INST_0_P0_POINT_S                       (1)
#define LWE397_SU_INST_0_P0_POINT_T                       (2)

#define LWE397_SU_INST_0_P1_LINE_WIDTH                      20:20
#define LWE397_SU_INST_0_P1_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_0_P1_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_0_P1_LINE_LENGTH                     21:21
#define LWE397_SU_INST_0_P1_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_0_P1_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_0_P1_POINT                   23:22
#define LWE397_SU_INST_0_P1_POINT_DISABLE                 (0)
#define LWE397_SU_INST_0_P1_POINT_S                       (1)
#define LWE397_SU_INST_0_P1_POINT_T                       (2)

#define LWE397_SU_INST_0_P2_LINE_WIDTH                      24:24
#define LWE397_SU_INST_0_P2_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_0_P2_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_0_P2_LINE_LENGTH                     25:25
#define LWE397_SU_INST_0_P2_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_0_P2_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_0_P2_POINT                   27:26
#define LWE397_SU_INST_0_P2_POINT_DISABLE                 (0)
#define LWE397_SU_INST_0_P2_POINT_S                       (1)
#define LWE397_SU_INST_0_P2_POINT_T                       (2)

#define LWE397_SU_INST_0_P3_LINE_WIDTH                      28:28
#define LWE397_SU_INST_0_P3_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_0_P3_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_0_P3_LINE_LENGTH                     29:29
#define LWE397_SU_INST_0_P3_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_0_P3_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_0_P3_POINT                   31:30
#define LWE397_SU_INST_0_P3_POINT_DISABLE                 (0)
#define LWE397_SU_INST_0_P3_POINT_S                       (1)
#define LWE397_SU_INST_0_P3_POINT_T                       (2)

#define LWE397_SU_INST_0_P0_TRAM_COL                        1:0

#define LWE397_SU_INST_0_P0_TRAM_FMT                        3:2
#define LWE397_SU_INST_0_P0_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_0_P0_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_0_P0_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_0_P0_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_0_P1_TRAM_COL                        5:4

#define LWE397_SU_INST_0_P1_TRAM_FMT                        7:6
#define LWE397_SU_INST_0_P1_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_0_P1_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_0_P1_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_0_P1_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_0_P2_TRAM_COL                        9:8

#define LWE397_SU_INST_0_P2_TRAM_FMT                        11:10
#define LWE397_SU_INST_0_P2_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_0_P2_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_0_P2_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_0_P2_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_0_P3_TRAM_COL                        13:12

#define LWE397_SU_INST_0_P3_TRAM_FMT                        15:14
#define LWE397_SU_INST_0_P3_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_0_P3_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_0_P3_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_0_P3_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_0_P0_TRI_SHADE_MODE                  16:16
#define LWE397_SU_INST_0_P0_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_0_P0_TRI_SHADE_MODE_FLAT                   (1)

#define LWE397_SU_INST_0_P1_TRI_SHADE_MODE                  17:17
#define LWE397_SU_INST_0_P1_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_0_P1_TRI_SHADE_MODE_FLAT                   (1)

#define LWE397_SU_INST_0_P2_TRI_SHADE_MODE                  18:18
#define LWE397_SU_INST_0_P2_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_0_P2_TRI_SHADE_MODE_FLAT                   (1)

#define LWE397_SU_INST_0_P3_TRI_SHADE_MODE                  19:19
#define LWE397_SU_INST_0_P3_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_0_P3_TRI_SHADE_MODE_FLAT                   (1)


// Register LWE397_SU_INST  
#define LWE397_SU_INST                    (0x300)
#define LWE397_SU_INST_SRC                  1:0
#define LWE397_SU_INST_SRC_VPE                    (0)
#define LWE397_SU_INST_SRC_Z                      (1)

#define LWE397_SU_INST_VC_ROW                       6:3

#define LWE397_SU_INST_TRAM_ROW                     14:9

#define LWE397_SU_INST_P0_LINE_WIDTH                        16:16
#define LWE397_SU_INST_P0_LINE_WIDTH_CONST                        (0)
#define LWE397_SU_INST_P0_LINE_WIDTH_VARYING                      (1)

#define LWE397_SU_INST_P0_LINE_LENGTH                       17:17
#define LWE397_SU_INST_P0_LINE_LENGTH_VARYING                     (0)
#define LWE397_SU_INST_P0_LINE_LENGTH_CONST                       (1)

#define LWE397_SU_INST_P0_POINT                     19:18
#define LWE397_SU_INST_P0_POINT_DISABLE                   (0)
#define LWE397_SU_INST_P0_POINT_S                 (1)
#define LWE397_SU_INST_P0_POINT_T                 (2)

#define LWE397_SU_INST_P1_LINE_WIDTH                        20:20
#define LWE397_SU_INST_P1_LINE_WIDTH_CONST                        (0)
#define LWE397_SU_INST_P1_LINE_WIDTH_VARYING                      (1)

#define LWE397_SU_INST_P1_LINE_LENGTH                       21:21
#define LWE397_SU_INST_P1_LINE_LENGTH_VARYING                     (0)
#define LWE397_SU_INST_P1_LINE_LENGTH_CONST                       (1)

#define LWE397_SU_INST_P1_POINT                     23:22
#define LWE397_SU_INST_P1_POINT_DISABLE                   (0)
#define LWE397_SU_INST_P1_POINT_S                 (1)
#define LWE397_SU_INST_P1_POINT_T                 (2)

#define LWE397_SU_INST_P2_LINE_WIDTH                        24:24
#define LWE397_SU_INST_P2_LINE_WIDTH_CONST                        (0)
#define LWE397_SU_INST_P2_LINE_WIDTH_VARYING                      (1)

#define LWE397_SU_INST_P2_LINE_LENGTH                       25:25
#define LWE397_SU_INST_P2_LINE_LENGTH_VARYING                     (0)
#define LWE397_SU_INST_P2_LINE_LENGTH_CONST                       (1)

#define LWE397_SU_INST_P2_POINT                     27:26
#define LWE397_SU_INST_P2_POINT_DISABLE                   (0)
#define LWE397_SU_INST_P2_POINT_S                 (1)
#define LWE397_SU_INST_P2_POINT_T                 (2)

#define LWE397_SU_INST_P3_LINE_WIDTH                        28:28
#define LWE397_SU_INST_P3_LINE_WIDTH_CONST                        (0)
#define LWE397_SU_INST_P3_LINE_WIDTH_VARYING                      (1)

#define LWE397_SU_INST_P3_LINE_LENGTH                       29:29
#define LWE397_SU_INST_P3_LINE_LENGTH_VARYING                     (0)
#define LWE397_SU_INST_P3_LINE_LENGTH_CONST                       (1)

#define LWE397_SU_INST_P3_POINT                     31:30
#define LWE397_SU_INST_P3_POINT_DISABLE                   (0)
#define LWE397_SU_INST_P3_POINT_S                 (1)
#define LWE397_SU_INST_P3_POINT_T                 (2)

#define LWE397_SU_INST_P0_TRAM_COL                  1:0

#define LWE397_SU_INST_P0_TRAM_FMT                  3:2
#define LWE397_SU_INST_P0_TRAM_FMT_NOP                    (0)
#define LWE397_SU_INST_P0_TRAM_FMT_LP_LO                  (1)
#define LWE397_SU_INST_P0_TRAM_FMT_LP_HI                  (2)
#define LWE397_SU_INST_P0_TRAM_FMT_HP                     (3)

#define LWE397_SU_INST_P1_TRAM_COL                  5:4

#define LWE397_SU_INST_P1_TRAM_FMT                  7:6
#define LWE397_SU_INST_P1_TRAM_FMT_NOP                    (0)
#define LWE397_SU_INST_P1_TRAM_FMT_LP_LO                  (1)
#define LWE397_SU_INST_P1_TRAM_FMT_LP_HI                  (2)
#define LWE397_SU_INST_P1_TRAM_FMT_HP                     (3)

#define LWE397_SU_INST_P2_TRAM_COL                  9:8

#define LWE397_SU_INST_P2_TRAM_FMT                  11:10
#define LWE397_SU_INST_P2_TRAM_FMT_NOP                    (0)
#define LWE397_SU_INST_P2_TRAM_FMT_LP_LO                  (1)
#define LWE397_SU_INST_P2_TRAM_FMT_LP_HI                  (2)
#define LWE397_SU_INST_P2_TRAM_FMT_HP                     (3)

#define LWE397_SU_INST_P3_TRAM_COL                  13:12

#define LWE397_SU_INST_P3_TRAM_FMT                  15:14
#define LWE397_SU_INST_P3_TRAM_FMT_NOP                    (0)
#define LWE397_SU_INST_P3_TRAM_FMT_LP_LO                  (1)
#define LWE397_SU_INST_P3_TRAM_FMT_LP_HI                  (2)
#define LWE397_SU_INST_P3_TRAM_FMT_HP                     (3)

#define LWE397_SU_INST_P0_TRI_SHADE_MODE                    16:16
#define LWE397_SU_INST_P0_TRI_SHADE_MODE_SMOOTH                   (0)
#define LWE397_SU_INST_P0_TRI_SHADE_MODE_FLAT                     (1)

#define LWE397_SU_INST_P1_TRI_SHADE_MODE                    17:17
#define LWE397_SU_INST_P1_TRI_SHADE_MODE_SMOOTH                   (0)
#define LWE397_SU_INST_P1_TRI_SHADE_MODE_FLAT                     (1)

#define LWE397_SU_INST_P2_TRI_SHADE_MODE                    18:18
#define LWE397_SU_INST_P2_TRI_SHADE_MODE_SMOOTH                   (0)
#define LWE397_SU_INST_P2_TRI_SHADE_MODE_FLAT                     (1)

#define LWE397_SU_INST_P3_TRI_SHADE_MODE                    19:19
#define LWE397_SU_INST_P3_TRI_SHADE_MODE_SMOOTH                   (0)
#define LWE397_SU_INST_P3_TRI_SHADE_MODE_FLAT                     (1)


// Register LWE397_SU_INST_1  
#define LWE397_SU_INST_1                  (0x301)
#define LWE397_SU_INST_1_SRC                        1:0
#define LWE397_SU_INST_1_SRC_VPE                  (0)
#define LWE397_SU_INST_1_SRC_Z                    (1)

#define LWE397_SU_INST_1_VC_ROW                     6:3

#define LWE397_SU_INST_1_TRAM_ROW                   14:9

#define LWE397_SU_INST_1_P0_LINE_WIDTH                      16:16
#define LWE397_SU_INST_1_P0_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_1_P0_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_1_P0_LINE_LENGTH                     17:17
#define LWE397_SU_INST_1_P0_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_1_P0_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_1_P0_POINT                   19:18
#define LWE397_SU_INST_1_P0_POINT_DISABLE                 (0)
#define LWE397_SU_INST_1_P0_POINT_S                       (1)
#define LWE397_SU_INST_1_P0_POINT_T                       (2)

#define LWE397_SU_INST_1_P1_LINE_WIDTH                      20:20
#define LWE397_SU_INST_1_P1_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_1_P1_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_1_P1_LINE_LENGTH                     21:21
#define LWE397_SU_INST_1_P1_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_1_P1_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_1_P1_POINT                   23:22
#define LWE397_SU_INST_1_P1_POINT_DISABLE                 (0)
#define LWE397_SU_INST_1_P1_POINT_S                       (1)
#define LWE397_SU_INST_1_P1_POINT_T                       (2)

#define LWE397_SU_INST_1_P2_LINE_WIDTH                      24:24
#define LWE397_SU_INST_1_P2_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_1_P2_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_1_P2_LINE_LENGTH                     25:25
#define LWE397_SU_INST_1_P2_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_1_P2_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_1_P2_POINT                   27:26
#define LWE397_SU_INST_1_P2_POINT_DISABLE                 (0)
#define LWE397_SU_INST_1_P2_POINT_S                       (1)
#define LWE397_SU_INST_1_P2_POINT_T                       (2)

#define LWE397_SU_INST_1_P3_LINE_WIDTH                      28:28
#define LWE397_SU_INST_1_P3_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_1_P3_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_1_P3_LINE_LENGTH                     29:29
#define LWE397_SU_INST_1_P3_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_1_P3_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_1_P3_POINT                   31:30
#define LWE397_SU_INST_1_P3_POINT_DISABLE                 (0)
#define LWE397_SU_INST_1_P3_POINT_S                       (1)
#define LWE397_SU_INST_1_P3_POINT_T                       (2)

#define LWE397_SU_INST_1_P0_TRAM_COL                        1:0

#define LWE397_SU_INST_1_P0_TRAM_FMT                        3:2
#define LWE397_SU_INST_1_P0_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_1_P0_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_1_P0_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_1_P0_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_1_P1_TRAM_COL                        5:4

#define LWE397_SU_INST_1_P1_TRAM_FMT                        7:6
#define LWE397_SU_INST_1_P1_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_1_P1_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_1_P1_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_1_P1_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_1_P2_TRAM_COL                        9:8

#define LWE397_SU_INST_1_P2_TRAM_FMT                        11:10
#define LWE397_SU_INST_1_P2_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_1_P2_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_1_P2_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_1_P2_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_1_P3_TRAM_COL                        13:12

#define LWE397_SU_INST_1_P3_TRAM_FMT                        15:14
#define LWE397_SU_INST_1_P3_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_1_P3_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_1_P3_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_1_P3_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_1_P0_TRI_SHADE_MODE                  16:16
#define LWE397_SU_INST_1_P0_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_1_P0_TRI_SHADE_MODE_FLAT                   (1)

#define LWE397_SU_INST_1_P1_TRI_SHADE_MODE                  17:17
#define LWE397_SU_INST_1_P1_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_1_P1_TRI_SHADE_MODE_FLAT                   (1)

#define LWE397_SU_INST_1_P2_TRI_SHADE_MODE                  18:18
#define LWE397_SU_INST_1_P2_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_1_P2_TRI_SHADE_MODE_FLAT                   (1)

#define LWE397_SU_INST_1_P3_TRI_SHADE_MODE                  19:19
#define LWE397_SU_INST_1_P3_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_1_P3_TRI_SHADE_MODE_FLAT                   (1)


// Register LWE397_SU_INST_2  
#define LWE397_SU_INST_2                  (0x302)
#define LWE397_SU_INST_2_SRC                        1:0
#define LWE397_SU_INST_2_SRC_VPE                  (0)
#define LWE397_SU_INST_2_SRC_Z                    (1)

#define LWE397_SU_INST_2_VC_ROW                     6:3

#define LWE397_SU_INST_2_TRAM_ROW                   14:9

#define LWE397_SU_INST_2_P0_LINE_WIDTH                      16:16
#define LWE397_SU_INST_2_P0_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_2_P0_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_2_P0_LINE_LENGTH                     17:17
#define LWE397_SU_INST_2_P0_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_2_P0_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_2_P0_POINT                   19:18
#define LWE397_SU_INST_2_P0_POINT_DISABLE                 (0)
#define LWE397_SU_INST_2_P0_POINT_S                       (1)
#define LWE397_SU_INST_2_P0_POINT_T                       (2)

#define LWE397_SU_INST_2_P1_LINE_WIDTH                      20:20
#define LWE397_SU_INST_2_P1_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_2_P1_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_2_P1_LINE_LENGTH                     21:21
#define LWE397_SU_INST_2_P1_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_2_P1_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_2_P1_POINT                   23:22
#define LWE397_SU_INST_2_P1_POINT_DISABLE                 (0)
#define LWE397_SU_INST_2_P1_POINT_S                       (1)
#define LWE397_SU_INST_2_P1_POINT_T                       (2)

#define LWE397_SU_INST_2_P2_LINE_WIDTH                      24:24
#define LWE397_SU_INST_2_P2_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_2_P2_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_2_P2_LINE_LENGTH                     25:25
#define LWE397_SU_INST_2_P2_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_2_P2_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_2_P2_POINT                   27:26
#define LWE397_SU_INST_2_P2_POINT_DISABLE                 (0)
#define LWE397_SU_INST_2_P2_POINT_S                       (1)
#define LWE397_SU_INST_2_P2_POINT_T                       (2)

#define LWE397_SU_INST_2_P3_LINE_WIDTH                      28:28
#define LWE397_SU_INST_2_P3_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_2_P3_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_2_P3_LINE_LENGTH                     29:29
#define LWE397_SU_INST_2_P3_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_2_P3_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_2_P3_POINT                   31:30
#define LWE397_SU_INST_2_P3_POINT_DISABLE                 (0)
#define LWE397_SU_INST_2_P3_POINT_S                       (1)
#define LWE397_SU_INST_2_P3_POINT_T                       (2)

#define LWE397_SU_INST_2_P0_TRAM_COL                        1:0

#define LWE397_SU_INST_2_P0_TRAM_FMT                        3:2
#define LWE397_SU_INST_2_P0_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_2_P0_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_2_P0_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_2_P0_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_2_P1_TRAM_COL                        5:4

#define LWE397_SU_INST_2_P1_TRAM_FMT                        7:6
#define LWE397_SU_INST_2_P1_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_2_P1_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_2_P1_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_2_P1_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_2_P2_TRAM_COL                        9:8

#define LWE397_SU_INST_2_P2_TRAM_FMT                        11:10
#define LWE397_SU_INST_2_P2_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_2_P2_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_2_P2_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_2_P2_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_2_P3_TRAM_COL                        13:12

#define LWE397_SU_INST_2_P3_TRAM_FMT                        15:14
#define LWE397_SU_INST_2_P3_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_2_P3_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_2_P3_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_2_P3_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_2_P0_TRI_SHADE_MODE                  16:16
#define LWE397_SU_INST_2_P0_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_2_P0_TRI_SHADE_MODE_FLAT                   (1)

#define LWE397_SU_INST_2_P1_TRI_SHADE_MODE                  17:17
#define LWE397_SU_INST_2_P1_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_2_P1_TRI_SHADE_MODE_FLAT                   (1)

#define LWE397_SU_INST_2_P2_TRI_SHADE_MODE                  18:18
#define LWE397_SU_INST_2_P2_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_2_P2_TRI_SHADE_MODE_FLAT                   (1)

#define LWE397_SU_INST_2_P3_TRI_SHADE_MODE                  19:19
#define LWE397_SU_INST_2_P3_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_2_P3_TRI_SHADE_MODE_FLAT                   (1)


// Register LWE397_SU_INST_3  
#define LWE397_SU_INST_3                  (0x303)
#define LWE397_SU_INST_3_SRC                        1:0
#define LWE397_SU_INST_3_SRC_VPE                  (0)
#define LWE397_SU_INST_3_SRC_Z                    (1)

#define LWE397_SU_INST_3_VC_ROW                     6:3

#define LWE397_SU_INST_3_TRAM_ROW                   14:9

#define LWE397_SU_INST_3_P0_LINE_WIDTH                      16:16
#define LWE397_SU_INST_3_P0_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_3_P0_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_3_P0_LINE_LENGTH                     17:17
#define LWE397_SU_INST_3_P0_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_3_P0_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_3_P0_POINT                   19:18
#define LWE397_SU_INST_3_P0_POINT_DISABLE                 (0)
#define LWE397_SU_INST_3_P0_POINT_S                       (1)
#define LWE397_SU_INST_3_P0_POINT_T                       (2)

#define LWE397_SU_INST_3_P1_LINE_WIDTH                      20:20
#define LWE397_SU_INST_3_P1_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_3_P1_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_3_P1_LINE_LENGTH                     21:21
#define LWE397_SU_INST_3_P1_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_3_P1_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_3_P1_POINT                   23:22
#define LWE397_SU_INST_3_P1_POINT_DISABLE                 (0)
#define LWE397_SU_INST_3_P1_POINT_S                       (1)
#define LWE397_SU_INST_3_P1_POINT_T                       (2)

#define LWE397_SU_INST_3_P2_LINE_WIDTH                      24:24
#define LWE397_SU_INST_3_P2_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_3_P2_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_3_P2_LINE_LENGTH                     25:25
#define LWE397_SU_INST_3_P2_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_3_P2_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_3_P2_POINT                   27:26
#define LWE397_SU_INST_3_P2_POINT_DISABLE                 (0)
#define LWE397_SU_INST_3_P2_POINT_S                       (1)
#define LWE397_SU_INST_3_P2_POINT_T                       (2)

#define LWE397_SU_INST_3_P3_LINE_WIDTH                      28:28
#define LWE397_SU_INST_3_P3_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_3_P3_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_3_P3_LINE_LENGTH                     29:29
#define LWE397_SU_INST_3_P3_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_3_P3_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_3_P3_POINT                   31:30
#define LWE397_SU_INST_3_P3_POINT_DISABLE                 (0)
#define LWE397_SU_INST_3_P3_POINT_S                       (1)
#define LWE397_SU_INST_3_P3_POINT_T                       (2)

#define LWE397_SU_INST_3_P0_TRAM_COL                        1:0

#define LWE397_SU_INST_3_P0_TRAM_FMT                        3:2
#define LWE397_SU_INST_3_P0_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_3_P0_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_3_P0_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_3_P0_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_3_P1_TRAM_COL                        5:4

#define LWE397_SU_INST_3_P1_TRAM_FMT                        7:6
#define LWE397_SU_INST_3_P1_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_3_P1_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_3_P1_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_3_P1_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_3_P2_TRAM_COL                        9:8

#define LWE397_SU_INST_3_P2_TRAM_FMT                        11:10
#define LWE397_SU_INST_3_P2_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_3_P2_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_3_P2_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_3_P2_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_3_P3_TRAM_COL                        13:12

#define LWE397_SU_INST_3_P3_TRAM_FMT                        15:14
#define LWE397_SU_INST_3_P3_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_3_P3_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_3_P3_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_3_P3_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_3_P0_TRI_SHADE_MODE                  16:16
#define LWE397_SU_INST_3_P0_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_3_P0_TRI_SHADE_MODE_FLAT                   (1)

#define LWE397_SU_INST_3_P1_TRI_SHADE_MODE                  17:17
#define LWE397_SU_INST_3_P1_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_3_P1_TRI_SHADE_MODE_FLAT                   (1)

#define LWE397_SU_INST_3_P2_TRI_SHADE_MODE                  18:18
#define LWE397_SU_INST_3_P2_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_3_P2_TRI_SHADE_MODE_FLAT                   (1)

#define LWE397_SU_INST_3_P3_TRI_SHADE_MODE                  19:19
#define LWE397_SU_INST_3_P3_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_3_P3_TRI_SHADE_MODE_FLAT                   (1)


// Register LWE397_SU_INST_4  
#define LWE397_SU_INST_4                  (0x304)
#define LWE397_SU_INST_4_SRC                        1:0
#define LWE397_SU_INST_4_SRC_VPE                  (0)
#define LWE397_SU_INST_4_SRC_Z                    (1)

#define LWE397_SU_INST_4_VC_ROW                     6:3

#define LWE397_SU_INST_4_TRAM_ROW                   14:9

#define LWE397_SU_INST_4_P0_LINE_WIDTH                      16:16
#define LWE397_SU_INST_4_P0_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_4_P0_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_4_P0_LINE_LENGTH                     17:17
#define LWE397_SU_INST_4_P0_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_4_P0_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_4_P0_POINT                   19:18
#define LWE397_SU_INST_4_P0_POINT_DISABLE                 (0)
#define LWE397_SU_INST_4_P0_POINT_S                       (1)
#define LWE397_SU_INST_4_P0_POINT_T                       (2)

#define LWE397_SU_INST_4_P1_LINE_WIDTH                      20:20
#define LWE397_SU_INST_4_P1_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_4_P1_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_4_P1_LINE_LENGTH                     21:21
#define LWE397_SU_INST_4_P1_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_4_P1_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_4_P1_POINT                   23:22
#define LWE397_SU_INST_4_P1_POINT_DISABLE                 (0)
#define LWE397_SU_INST_4_P1_POINT_S                       (1)
#define LWE397_SU_INST_4_P1_POINT_T                       (2)

#define LWE397_SU_INST_4_P2_LINE_WIDTH                      24:24
#define LWE397_SU_INST_4_P2_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_4_P2_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_4_P2_LINE_LENGTH                     25:25
#define LWE397_SU_INST_4_P2_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_4_P2_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_4_P2_POINT                   27:26
#define LWE397_SU_INST_4_P2_POINT_DISABLE                 (0)
#define LWE397_SU_INST_4_P2_POINT_S                       (1)
#define LWE397_SU_INST_4_P2_POINT_T                       (2)

#define LWE397_SU_INST_4_P3_LINE_WIDTH                      28:28
#define LWE397_SU_INST_4_P3_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_4_P3_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_4_P3_LINE_LENGTH                     29:29
#define LWE397_SU_INST_4_P3_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_4_P3_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_4_P3_POINT                   31:30
#define LWE397_SU_INST_4_P3_POINT_DISABLE                 (0)
#define LWE397_SU_INST_4_P3_POINT_S                       (1)
#define LWE397_SU_INST_4_P3_POINT_T                       (2)

#define LWE397_SU_INST_4_P0_TRAM_COL                        1:0

#define LWE397_SU_INST_4_P0_TRAM_FMT                        3:2
#define LWE397_SU_INST_4_P0_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_4_P0_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_4_P0_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_4_P0_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_4_P1_TRAM_COL                        5:4

#define LWE397_SU_INST_4_P1_TRAM_FMT                        7:6
#define LWE397_SU_INST_4_P1_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_4_P1_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_4_P1_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_4_P1_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_4_P2_TRAM_COL                        9:8

#define LWE397_SU_INST_4_P2_TRAM_FMT                        11:10
#define LWE397_SU_INST_4_P2_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_4_P2_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_4_P2_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_4_P2_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_4_P3_TRAM_COL                        13:12

#define LWE397_SU_INST_4_P3_TRAM_FMT                        15:14
#define LWE397_SU_INST_4_P3_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_4_P3_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_4_P3_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_4_P3_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_4_P0_TRI_SHADE_MODE                  16:16
#define LWE397_SU_INST_4_P0_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_4_P0_TRI_SHADE_MODE_FLAT                   (1)

#define LWE397_SU_INST_4_P1_TRI_SHADE_MODE                  17:17
#define LWE397_SU_INST_4_P1_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_4_P1_TRI_SHADE_MODE_FLAT                   (1)

#define LWE397_SU_INST_4_P2_TRI_SHADE_MODE                  18:18
#define LWE397_SU_INST_4_P2_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_4_P2_TRI_SHADE_MODE_FLAT                   (1)

#define LWE397_SU_INST_4_P3_TRI_SHADE_MODE                  19:19
#define LWE397_SU_INST_4_P3_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_4_P3_TRI_SHADE_MODE_FLAT                   (1)


// Register LWE397_SU_INST_5  
#define LWE397_SU_INST_5                  (0x305)
#define LWE397_SU_INST_5_SRC                        1:0
#define LWE397_SU_INST_5_SRC_VPE                  (0)
#define LWE397_SU_INST_5_SRC_Z                    (1)

#define LWE397_SU_INST_5_VC_ROW                     6:3

#define LWE397_SU_INST_5_TRAM_ROW                   14:9

#define LWE397_SU_INST_5_P0_LINE_WIDTH                      16:16
#define LWE397_SU_INST_5_P0_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_5_P0_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_5_P0_LINE_LENGTH                     17:17
#define LWE397_SU_INST_5_P0_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_5_P0_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_5_P0_POINT                   19:18
#define LWE397_SU_INST_5_P0_POINT_DISABLE                 (0)
#define LWE397_SU_INST_5_P0_POINT_S                       (1)
#define LWE397_SU_INST_5_P0_POINT_T                       (2)

#define LWE397_SU_INST_5_P1_LINE_WIDTH                      20:20
#define LWE397_SU_INST_5_P1_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_5_P1_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_5_P1_LINE_LENGTH                     21:21
#define LWE397_SU_INST_5_P1_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_5_P1_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_5_P1_POINT                   23:22
#define LWE397_SU_INST_5_P1_POINT_DISABLE                 (0)
#define LWE397_SU_INST_5_P1_POINT_S                       (1)
#define LWE397_SU_INST_5_P1_POINT_T                       (2)

#define LWE397_SU_INST_5_P2_LINE_WIDTH                      24:24
#define LWE397_SU_INST_5_P2_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_5_P2_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_5_P2_LINE_LENGTH                     25:25
#define LWE397_SU_INST_5_P2_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_5_P2_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_5_P2_POINT                   27:26
#define LWE397_SU_INST_5_P2_POINT_DISABLE                 (0)
#define LWE397_SU_INST_5_P2_POINT_S                       (1)
#define LWE397_SU_INST_5_P2_POINT_T                       (2)

#define LWE397_SU_INST_5_P3_LINE_WIDTH                      28:28
#define LWE397_SU_INST_5_P3_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_5_P3_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_5_P3_LINE_LENGTH                     29:29
#define LWE397_SU_INST_5_P3_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_5_P3_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_5_P3_POINT                   31:30
#define LWE397_SU_INST_5_P3_POINT_DISABLE                 (0)
#define LWE397_SU_INST_5_P3_POINT_S                       (1)
#define LWE397_SU_INST_5_P3_POINT_T                       (2)

#define LWE397_SU_INST_5_P0_TRAM_COL                        1:0

#define LWE397_SU_INST_5_P0_TRAM_FMT                        3:2
#define LWE397_SU_INST_5_P0_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_5_P0_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_5_P0_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_5_P0_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_5_P1_TRAM_COL                        5:4

#define LWE397_SU_INST_5_P1_TRAM_FMT                        7:6
#define LWE397_SU_INST_5_P1_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_5_P1_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_5_P1_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_5_P1_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_5_P2_TRAM_COL                        9:8

#define LWE397_SU_INST_5_P2_TRAM_FMT                        11:10
#define LWE397_SU_INST_5_P2_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_5_P2_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_5_P2_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_5_P2_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_5_P3_TRAM_COL                        13:12

#define LWE397_SU_INST_5_P3_TRAM_FMT                        15:14
#define LWE397_SU_INST_5_P3_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_5_P3_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_5_P3_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_5_P3_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_5_P0_TRI_SHADE_MODE                  16:16
#define LWE397_SU_INST_5_P0_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_5_P0_TRI_SHADE_MODE_FLAT                   (1)

#define LWE397_SU_INST_5_P1_TRI_SHADE_MODE                  17:17
#define LWE397_SU_INST_5_P1_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_5_P1_TRI_SHADE_MODE_FLAT                   (1)

#define LWE397_SU_INST_5_P2_TRI_SHADE_MODE                  18:18
#define LWE397_SU_INST_5_P2_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_5_P2_TRI_SHADE_MODE_FLAT                   (1)

#define LWE397_SU_INST_5_P3_TRI_SHADE_MODE                  19:19
#define LWE397_SU_INST_5_P3_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_5_P3_TRI_SHADE_MODE_FLAT                   (1)


// Register LWE397_SU_INST_6  
#define LWE397_SU_INST_6                  (0x306)
#define LWE397_SU_INST_6_SRC                        1:0
#define LWE397_SU_INST_6_SRC_VPE                  (0)
#define LWE397_SU_INST_6_SRC_Z                    (1)

#define LWE397_SU_INST_6_VC_ROW                     6:3

#define LWE397_SU_INST_6_TRAM_ROW                   14:9

#define LWE397_SU_INST_6_P0_LINE_WIDTH                      16:16
#define LWE397_SU_INST_6_P0_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_6_P0_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_6_P0_LINE_LENGTH                     17:17
#define LWE397_SU_INST_6_P0_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_6_P0_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_6_P0_POINT                   19:18
#define LWE397_SU_INST_6_P0_POINT_DISABLE                 (0)
#define LWE397_SU_INST_6_P0_POINT_S                       (1)
#define LWE397_SU_INST_6_P0_POINT_T                       (2)

#define LWE397_SU_INST_6_P1_LINE_WIDTH                      20:20
#define LWE397_SU_INST_6_P1_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_6_P1_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_6_P1_LINE_LENGTH                     21:21
#define LWE397_SU_INST_6_P1_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_6_P1_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_6_P1_POINT                   23:22
#define LWE397_SU_INST_6_P1_POINT_DISABLE                 (0)
#define LWE397_SU_INST_6_P1_POINT_S                       (1)
#define LWE397_SU_INST_6_P1_POINT_T                       (2)

#define LWE397_SU_INST_6_P2_LINE_WIDTH                      24:24
#define LWE397_SU_INST_6_P2_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_6_P2_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_6_P2_LINE_LENGTH                     25:25
#define LWE397_SU_INST_6_P2_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_6_P2_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_6_P2_POINT                   27:26
#define LWE397_SU_INST_6_P2_POINT_DISABLE                 (0)
#define LWE397_SU_INST_6_P2_POINT_S                       (1)
#define LWE397_SU_INST_6_P2_POINT_T                       (2)

#define LWE397_SU_INST_6_P3_LINE_WIDTH                      28:28
#define LWE397_SU_INST_6_P3_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_6_P3_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_6_P3_LINE_LENGTH                     29:29
#define LWE397_SU_INST_6_P3_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_6_P3_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_6_P3_POINT                   31:30
#define LWE397_SU_INST_6_P3_POINT_DISABLE                 (0)
#define LWE397_SU_INST_6_P3_POINT_S                       (1)
#define LWE397_SU_INST_6_P3_POINT_T                       (2)

#define LWE397_SU_INST_6_P0_TRAM_COL                        1:0

#define LWE397_SU_INST_6_P0_TRAM_FMT                        3:2
#define LWE397_SU_INST_6_P0_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_6_P0_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_6_P0_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_6_P0_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_6_P1_TRAM_COL                        5:4

#define LWE397_SU_INST_6_P1_TRAM_FMT                        7:6
#define LWE397_SU_INST_6_P1_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_6_P1_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_6_P1_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_6_P1_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_6_P2_TRAM_COL                        9:8

#define LWE397_SU_INST_6_P2_TRAM_FMT                        11:10
#define LWE397_SU_INST_6_P2_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_6_P2_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_6_P2_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_6_P2_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_6_P3_TRAM_COL                        13:12

#define LWE397_SU_INST_6_P3_TRAM_FMT                        15:14
#define LWE397_SU_INST_6_P3_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_6_P3_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_6_P3_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_6_P3_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_6_P0_TRI_SHADE_MODE                  16:16
#define LWE397_SU_INST_6_P0_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_6_P0_TRI_SHADE_MODE_FLAT                   (1)

#define LWE397_SU_INST_6_P1_TRI_SHADE_MODE                  17:17
#define LWE397_SU_INST_6_P1_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_6_P1_TRI_SHADE_MODE_FLAT                   (1)

#define LWE397_SU_INST_6_P2_TRI_SHADE_MODE                  18:18
#define LWE397_SU_INST_6_P2_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_6_P2_TRI_SHADE_MODE_FLAT                   (1)

#define LWE397_SU_INST_6_P3_TRI_SHADE_MODE                  19:19
#define LWE397_SU_INST_6_P3_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_6_P3_TRI_SHADE_MODE_FLAT                   (1)


// Register LWE397_SU_INST_7  
#define LWE397_SU_INST_7                  (0x307)
#define LWE397_SU_INST_7_SRC                        1:0
#define LWE397_SU_INST_7_SRC_VPE                  (0)
#define LWE397_SU_INST_7_SRC_Z                    (1)

#define LWE397_SU_INST_7_VC_ROW                     6:3

#define LWE397_SU_INST_7_TRAM_ROW                   14:9

#define LWE397_SU_INST_7_P0_LINE_WIDTH                      16:16
#define LWE397_SU_INST_7_P0_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_7_P0_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_7_P0_LINE_LENGTH                     17:17
#define LWE397_SU_INST_7_P0_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_7_P0_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_7_P0_POINT                   19:18
#define LWE397_SU_INST_7_P0_POINT_DISABLE                 (0)
#define LWE397_SU_INST_7_P0_POINT_S                       (1)
#define LWE397_SU_INST_7_P0_POINT_T                       (2)

#define LWE397_SU_INST_7_P1_LINE_WIDTH                      20:20
#define LWE397_SU_INST_7_P1_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_7_P1_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_7_P1_LINE_LENGTH                     21:21
#define LWE397_SU_INST_7_P1_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_7_P1_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_7_P1_POINT                   23:22
#define LWE397_SU_INST_7_P1_POINT_DISABLE                 (0)
#define LWE397_SU_INST_7_P1_POINT_S                       (1)
#define LWE397_SU_INST_7_P1_POINT_T                       (2)

#define LWE397_SU_INST_7_P2_LINE_WIDTH                      24:24
#define LWE397_SU_INST_7_P2_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_7_P2_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_7_P2_LINE_LENGTH                     25:25
#define LWE397_SU_INST_7_P2_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_7_P2_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_7_P2_POINT                   27:26
#define LWE397_SU_INST_7_P2_POINT_DISABLE                 (0)
#define LWE397_SU_INST_7_P2_POINT_S                       (1)
#define LWE397_SU_INST_7_P2_POINT_T                       (2)

#define LWE397_SU_INST_7_P3_LINE_WIDTH                      28:28
#define LWE397_SU_INST_7_P3_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_7_P3_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_7_P3_LINE_LENGTH                     29:29
#define LWE397_SU_INST_7_P3_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_7_P3_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_7_P3_POINT                   31:30
#define LWE397_SU_INST_7_P3_POINT_DISABLE                 (0)
#define LWE397_SU_INST_7_P3_POINT_S                       (1)
#define LWE397_SU_INST_7_P3_POINT_T                       (2)

#define LWE397_SU_INST_7_P0_TRAM_COL                        1:0

#define LWE397_SU_INST_7_P0_TRAM_FMT                        3:2
#define LWE397_SU_INST_7_P0_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_7_P0_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_7_P0_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_7_P0_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_7_P1_TRAM_COL                        5:4

#define LWE397_SU_INST_7_P1_TRAM_FMT                        7:6
#define LWE397_SU_INST_7_P1_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_7_P1_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_7_P1_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_7_P1_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_7_P2_TRAM_COL                        9:8

#define LWE397_SU_INST_7_P2_TRAM_FMT                        11:10
#define LWE397_SU_INST_7_P2_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_7_P2_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_7_P2_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_7_P2_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_7_P3_TRAM_COL                        13:12

#define LWE397_SU_INST_7_P3_TRAM_FMT                        15:14
#define LWE397_SU_INST_7_P3_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_7_P3_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_7_P3_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_7_P3_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_7_P0_TRI_SHADE_MODE                  16:16
#define LWE397_SU_INST_7_P0_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_7_P0_TRI_SHADE_MODE_FLAT                   (1)

#define LWE397_SU_INST_7_P1_TRI_SHADE_MODE                  17:17
#define LWE397_SU_INST_7_P1_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_7_P1_TRI_SHADE_MODE_FLAT                   (1)

#define LWE397_SU_INST_7_P2_TRI_SHADE_MODE                  18:18
#define LWE397_SU_INST_7_P2_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_7_P2_TRI_SHADE_MODE_FLAT                   (1)

#define LWE397_SU_INST_7_P3_TRI_SHADE_MODE                  19:19
#define LWE397_SU_INST_7_P3_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_7_P3_TRI_SHADE_MODE_FLAT                   (1)


// Register LWE397_SU_INST_8  
#define LWE397_SU_INST_8                  (0x308)
#define LWE397_SU_INST_8_SRC                        1:0
#define LWE397_SU_INST_8_SRC_VPE                  (0)
#define LWE397_SU_INST_8_SRC_Z                    (1)

#define LWE397_SU_INST_8_VC_ROW                     6:3

#define LWE397_SU_INST_8_TRAM_ROW                   14:9

#define LWE397_SU_INST_8_P0_LINE_WIDTH                      16:16
#define LWE397_SU_INST_8_P0_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_8_P0_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_8_P0_LINE_LENGTH                     17:17
#define LWE397_SU_INST_8_P0_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_8_P0_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_8_P0_POINT                   19:18
#define LWE397_SU_INST_8_P0_POINT_DISABLE                 (0)
#define LWE397_SU_INST_8_P0_POINT_S                       (1)
#define LWE397_SU_INST_8_P0_POINT_T                       (2)

#define LWE397_SU_INST_8_P1_LINE_WIDTH                      20:20
#define LWE397_SU_INST_8_P1_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_8_P1_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_8_P1_LINE_LENGTH                     21:21
#define LWE397_SU_INST_8_P1_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_8_P1_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_8_P1_POINT                   23:22
#define LWE397_SU_INST_8_P1_POINT_DISABLE                 (0)
#define LWE397_SU_INST_8_P1_POINT_S                       (1)
#define LWE397_SU_INST_8_P1_POINT_T                       (2)

#define LWE397_SU_INST_8_P2_LINE_WIDTH                      24:24
#define LWE397_SU_INST_8_P2_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_8_P2_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_8_P2_LINE_LENGTH                     25:25
#define LWE397_SU_INST_8_P2_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_8_P2_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_8_P2_POINT                   27:26
#define LWE397_SU_INST_8_P2_POINT_DISABLE                 (0)
#define LWE397_SU_INST_8_P2_POINT_S                       (1)
#define LWE397_SU_INST_8_P2_POINT_T                       (2)

#define LWE397_SU_INST_8_P3_LINE_WIDTH                      28:28
#define LWE397_SU_INST_8_P3_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_8_P3_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_8_P3_LINE_LENGTH                     29:29
#define LWE397_SU_INST_8_P3_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_8_P3_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_8_P3_POINT                   31:30
#define LWE397_SU_INST_8_P3_POINT_DISABLE                 (0)
#define LWE397_SU_INST_8_P3_POINT_S                       (1)
#define LWE397_SU_INST_8_P3_POINT_T                       (2)

#define LWE397_SU_INST_8_P0_TRAM_COL                        1:0

#define LWE397_SU_INST_8_P0_TRAM_FMT                        3:2
#define LWE397_SU_INST_8_P0_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_8_P0_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_8_P0_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_8_P0_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_8_P1_TRAM_COL                        5:4

#define LWE397_SU_INST_8_P1_TRAM_FMT                        7:6
#define LWE397_SU_INST_8_P1_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_8_P1_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_8_P1_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_8_P1_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_8_P2_TRAM_COL                        9:8

#define LWE397_SU_INST_8_P2_TRAM_FMT                        11:10
#define LWE397_SU_INST_8_P2_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_8_P2_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_8_P2_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_8_P2_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_8_P3_TRAM_COL                        13:12

#define LWE397_SU_INST_8_P3_TRAM_FMT                        15:14
#define LWE397_SU_INST_8_P3_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_8_P3_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_8_P3_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_8_P3_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_8_P0_TRI_SHADE_MODE                  16:16
#define LWE397_SU_INST_8_P0_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_8_P0_TRI_SHADE_MODE_FLAT                   (1)

#define LWE397_SU_INST_8_P1_TRI_SHADE_MODE                  17:17
#define LWE397_SU_INST_8_P1_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_8_P1_TRI_SHADE_MODE_FLAT                   (1)

#define LWE397_SU_INST_8_P2_TRI_SHADE_MODE                  18:18
#define LWE397_SU_INST_8_P2_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_8_P2_TRI_SHADE_MODE_FLAT                   (1)

#define LWE397_SU_INST_8_P3_TRI_SHADE_MODE                  19:19
#define LWE397_SU_INST_8_P3_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_8_P3_TRI_SHADE_MODE_FLAT                   (1)


// Register LWE397_SU_INST_9  
#define LWE397_SU_INST_9                  (0x309)
#define LWE397_SU_INST_9_SRC                        1:0
#define LWE397_SU_INST_9_SRC_VPE                  (0)
#define LWE397_SU_INST_9_SRC_Z                    (1)

#define LWE397_SU_INST_9_VC_ROW                     6:3

#define LWE397_SU_INST_9_TRAM_ROW                   14:9

#define LWE397_SU_INST_9_P0_LINE_WIDTH                      16:16
#define LWE397_SU_INST_9_P0_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_9_P0_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_9_P0_LINE_LENGTH                     17:17
#define LWE397_SU_INST_9_P0_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_9_P0_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_9_P0_POINT                   19:18
#define LWE397_SU_INST_9_P0_POINT_DISABLE                 (0)
#define LWE397_SU_INST_9_P0_POINT_S                       (1)
#define LWE397_SU_INST_9_P0_POINT_T                       (2)

#define LWE397_SU_INST_9_P1_LINE_WIDTH                      20:20
#define LWE397_SU_INST_9_P1_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_9_P1_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_9_P1_LINE_LENGTH                     21:21
#define LWE397_SU_INST_9_P1_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_9_P1_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_9_P1_POINT                   23:22
#define LWE397_SU_INST_9_P1_POINT_DISABLE                 (0)
#define LWE397_SU_INST_9_P1_POINT_S                       (1)
#define LWE397_SU_INST_9_P1_POINT_T                       (2)

#define LWE397_SU_INST_9_P2_LINE_WIDTH                      24:24
#define LWE397_SU_INST_9_P2_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_9_P2_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_9_P2_LINE_LENGTH                     25:25
#define LWE397_SU_INST_9_P2_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_9_P2_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_9_P2_POINT                   27:26
#define LWE397_SU_INST_9_P2_POINT_DISABLE                 (0)
#define LWE397_SU_INST_9_P2_POINT_S                       (1)
#define LWE397_SU_INST_9_P2_POINT_T                       (2)

#define LWE397_SU_INST_9_P3_LINE_WIDTH                      28:28
#define LWE397_SU_INST_9_P3_LINE_WIDTH_CONST                      (0)
#define LWE397_SU_INST_9_P3_LINE_WIDTH_VARYING                    (1)

#define LWE397_SU_INST_9_P3_LINE_LENGTH                     29:29
#define LWE397_SU_INST_9_P3_LINE_LENGTH_VARYING                   (0)
#define LWE397_SU_INST_9_P3_LINE_LENGTH_CONST                     (1)

#define LWE397_SU_INST_9_P3_POINT                   31:30
#define LWE397_SU_INST_9_P3_POINT_DISABLE                 (0)
#define LWE397_SU_INST_9_P3_POINT_S                       (1)
#define LWE397_SU_INST_9_P3_POINT_T                       (2)

#define LWE397_SU_INST_9_P0_TRAM_COL                        1:0

#define LWE397_SU_INST_9_P0_TRAM_FMT                        3:2
#define LWE397_SU_INST_9_P0_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_9_P0_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_9_P0_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_9_P0_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_9_P1_TRAM_COL                        5:4

#define LWE397_SU_INST_9_P1_TRAM_FMT                        7:6
#define LWE397_SU_INST_9_P1_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_9_P1_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_9_P1_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_9_P1_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_9_P2_TRAM_COL                        9:8

#define LWE397_SU_INST_9_P2_TRAM_FMT                        11:10
#define LWE397_SU_INST_9_P2_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_9_P2_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_9_P2_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_9_P2_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_9_P3_TRAM_COL                        13:12

#define LWE397_SU_INST_9_P3_TRAM_FMT                        15:14
#define LWE397_SU_INST_9_P3_TRAM_FMT_NOP                  (0)
#define LWE397_SU_INST_9_P3_TRAM_FMT_LP_LO                        (1)
#define LWE397_SU_INST_9_P3_TRAM_FMT_LP_HI                        (2)
#define LWE397_SU_INST_9_P3_TRAM_FMT_HP                   (3)

#define LWE397_SU_INST_9_P0_TRI_SHADE_MODE                  16:16
#define LWE397_SU_INST_9_P0_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_9_P0_TRI_SHADE_MODE_FLAT                   (1)

#define LWE397_SU_INST_9_P1_TRI_SHADE_MODE                  17:17
#define LWE397_SU_INST_9_P1_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_9_P1_TRI_SHADE_MODE_FLAT                   (1)

#define LWE397_SU_INST_9_P2_TRI_SHADE_MODE                  18:18
#define LWE397_SU_INST_9_P2_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_9_P2_TRI_SHADE_MODE_FLAT                   (1)

#define LWE397_SU_INST_9_P3_TRI_SHADE_MODE                  19:19
#define LWE397_SU_INST_9_P3_TRI_SHADE_MODE_SMOOTH                 (0)
#define LWE397_SU_INST_9_P3_TRI_SHADE_MODE_FLAT                   (1)


// Register LWE397_SU_INST_10  
#define LWE397_SU_INST_10                 (0x30a)
#define LWE397_SU_INST_10_SRC                       1:0
#define LWE397_SU_INST_10_SRC_VPE                 (0)
#define LWE397_SU_INST_10_SRC_Z                   (1)

#define LWE397_SU_INST_10_VC_ROW                    6:3

#define LWE397_SU_INST_10_TRAM_ROW                  14:9

#define LWE397_SU_INST_10_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_10_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_10_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_10_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_10_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_10_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_10_P0_POINT                  19:18
#define LWE397_SU_INST_10_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_10_P0_POINT_S                      (1)
#define LWE397_SU_INST_10_P0_POINT_T                      (2)

#define LWE397_SU_INST_10_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_10_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_10_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_10_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_10_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_10_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_10_P1_POINT                  23:22
#define LWE397_SU_INST_10_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_10_P1_POINT_S                      (1)
#define LWE397_SU_INST_10_P1_POINT_T                      (2)

#define LWE397_SU_INST_10_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_10_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_10_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_10_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_10_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_10_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_10_P2_POINT                  27:26
#define LWE397_SU_INST_10_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_10_P2_POINT_S                      (1)
#define LWE397_SU_INST_10_P2_POINT_T                      (2)

#define LWE397_SU_INST_10_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_10_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_10_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_10_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_10_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_10_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_10_P3_POINT                  31:30
#define LWE397_SU_INST_10_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_10_P3_POINT_S                      (1)
#define LWE397_SU_INST_10_P3_POINT_T                      (2)

#define LWE397_SU_INST_10_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_10_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_10_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_10_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_10_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_10_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_10_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_10_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_10_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_10_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_10_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_10_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_10_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_10_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_10_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_10_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_10_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_10_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_10_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_10_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_10_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_10_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_10_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_10_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_10_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_10_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_10_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_10_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_10_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_10_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_10_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_10_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_10_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_10_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_10_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_10_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_11  
#define LWE397_SU_INST_11                 (0x30b)
#define LWE397_SU_INST_11_SRC                       1:0
#define LWE397_SU_INST_11_SRC_VPE                 (0)
#define LWE397_SU_INST_11_SRC_Z                   (1)

#define LWE397_SU_INST_11_VC_ROW                    6:3

#define LWE397_SU_INST_11_TRAM_ROW                  14:9

#define LWE397_SU_INST_11_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_11_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_11_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_11_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_11_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_11_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_11_P0_POINT                  19:18
#define LWE397_SU_INST_11_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_11_P0_POINT_S                      (1)
#define LWE397_SU_INST_11_P0_POINT_T                      (2)

#define LWE397_SU_INST_11_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_11_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_11_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_11_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_11_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_11_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_11_P1_POINT                  23:22
#define LWE397_SU_INST_11_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_11_P1_POINT_S                      (1)
#define LWE397_SU_INST_11_P1_POINT_T                      (2)

#define LWE397_SU_INST_11_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_11_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_11_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_11_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_11_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_11_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_11_P2_POINT                  27:26
#define LWE397_SU_INST_11_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_11_P2_POINT_S                      (1)
#define LWE397_SU_INST_11_P2_POINT_T                      (2)

#define LWE397_SU_INST_11_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_11_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_11_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_11_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_11_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_11_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_11_P3_POINT                  31:30
#define LWE397_SU_INST_11_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_11_P3_POINT_S                      (1)
#define LWE397_SU_INST_11_P3_POINT_T                      (2)

#define LWE397_SU_INST_11_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_11_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_11_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_11_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_11_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_11_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_11_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_11_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_11_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_11_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_11_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_11_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_11_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_11_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_11_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_11_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_11_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_11_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_11_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_11_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_11_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_11_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_11_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_11_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_11_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_11_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_11_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_11_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_11_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_11_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_11_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_11_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_11_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_11_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_11_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_11_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_12  
#define LWE397_SU_INST_12                 (0x30c)
#define LWE397_SU_INST_12_SRC                       1:0
#define LWE397_SU_INST_12_SRC_VPE                 (0)
#define LWE397_SU_INST_12_SRC_Z                   (1)

#define LWE397_SU_INST_12_VC_ROW                    6:3

#define LWE397_SU_INST_12_TRAM_ROW                  14:9

#define LWE397_SU_INST_12_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_12_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_12_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_12_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_12_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_12_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_12_P0_POINT                  19:18
#define LWE397_SU_INST_12_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_12_P0_POINT_S                      (1)
#define LWE397_SU_INST_12_P0_POINT_T                      (2)

#define LWE397_SU_INST_12_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_12_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_12_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_12_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_12_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_12_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_12_P1_POINT                  23:22
#define LWE397_SU_INST_12_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_12_P1_POINT_S                      (1)
#define LWE397_SU_INST_12_P1_POINT_T                      (2)

#define LWE397_SU_INST_12_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_12_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_12_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_12_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_12_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_12_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_12_P2_POINT                  27:26
#define LWE397_SU_INST_12_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_12_P2_POINT_S                      (1)
#define LWE397_SU_INST_12_P2_POINT_T                      (2)

#define LWE397_SU_INST_12_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_12_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_12_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_12_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_12_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_12_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_12_P3_POINT                  31:30
#define LWE397_SU_INST_12_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_12_P3_POINT_S                      (1)
#define LWE397_SU_INST_12_P3_POINT_T                      (2)

#define LWE397_SU_INST_12_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_12_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_12_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_12_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_12_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_12_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_12_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_12_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_12_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_12_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_12_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_12_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_12_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_12_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_12_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_12_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_12_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_12_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_12_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_12_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_12_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_12_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_12_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_12_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_12_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_12_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_12_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_12_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_12_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_12_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_12_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_12_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_12_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_12_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_12_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_12_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_13  
#define LWE397_SU_INST_13                 (0x30d)
#define LWE397_SU_INST_13_SRC                       1:0
#define LWE397_SU_INST_13_SRC_VPE                 (0)
#define LWE397_SU_INST_13_SRC_Z                   (1)

#define LWE397_SU_INST_13_VC_ROW                    6:3

#define LWE397_SU_INST_13_TRAM_ROW                  14:9

#define LWE397_SU_INST_13_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_13_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_13_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_13_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_13_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_13_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_13_P0_POINT                  19:18
#define LWE397_SU_INST_13_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_13_P0_POINT_S                      (1)
#define LWE397_SU_INST_13_P0_POINT_T                      (2)

#define LWE397_SU_INST_13_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_13_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_13_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_13_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_13_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_13_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_13_P1_POINT                  23:22
#define LWE397_SU_INST_13_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_13_P1_POINT_S                      (1)
#define LWE397_SU_INST_13_P1_POINT_T                      (2)

#define LWE397_SU_INST_13_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_13_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_13_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_13_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_13_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_13_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_13_P2_POINT                  27:26
#define LWE397_SU_INST_13_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_13_P2_POINT_S                      (1)
#define LWE397_SU_INST_13_P2_POINT_T                      (2)

#define LWE397_SU_INST_13_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_13_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_13_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_13_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_13_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_13_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_13_P3_POINT                  31:30
#define LWE397_SU_INST_13_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_13_P3_POINT_S                      (1)
#define LWE397_SU_INST_13_P3_POINT_T                      (2)

#define LWE397_SU_INST_13_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_13_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_13_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_13_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_13_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_13_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_13_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_13_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_13_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_13_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_13_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_13_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_13_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_13_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_13_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_13_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_13_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_13_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_13_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_13_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_13_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_13_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_13_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_13_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_13_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_13_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_13_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_13_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_13_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_13_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_13_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_13_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_13_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_13_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_13_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_13_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_14  
#define LWE397_SU_INST_14                 (0x30e)
#define LWE397_SU_INST_14_SRC                       1:0
#define LWE397_SU_INST_14_SRC_VPE                 (0)
#define LWE397_SU_INST_14_SRC_Z                   (1)

#define LWE397_SU_INST_14_VC_ROW                    6:3

#define LWE397_SU_INST_14_TRAM_ROW                  14:9

#define LWE397_SU_INST_14_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_14_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_14_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_14_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_14_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_14_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_14_P0_POINT                  19:18
#define LWE397_SU_INST_14_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_14_P0_POINT_S                      (1)
#define LWE397_SU_INST_14_P0_POINT_T                      (2)

#define LWE397_SU_INST_14_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_14_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_14_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_14_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_14_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_14_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_14_P1_POINT                  23:22
#define LWE397_SU_INST_14_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_14_P1_POINT_S                      (1)
#define LWE397_SU_INST_14_P1_POINT_T                      (2)

#define LWE397_SU_INST_14_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_14_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_14_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_14_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_14_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_14_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_14_P2_POINT                  27:26
#define LWE397_SU_INST_14_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_14_P2_POINT_S                      (1)
#define LWE397_SU_INST_14_P2_POINT_T                      (2)

#define LWE397_SU_INST_14_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_14_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_14_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_14_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_14_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_14_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_14_P3_POINT                  31:30
#define LWE397_SU_INST_14_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_14_P3_POINT_S                      (1)
#define LWE397_SU_INST_14_P3_POINT_T                      (2)

#define LWE397_SU_INST_14_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_14_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_14_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_14_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_14_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_14_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_14_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_14_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_14_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_14_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_14_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_14_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_14_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_14_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_14_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_14_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_14_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_14_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_14_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_14_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_14_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_14_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_14_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_14_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_14_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_14_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_14_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_14_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_14_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_14_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_14_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_14_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_14_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_14_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_14_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_14_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_15  
#define LWE397_SU_INST_15                 (0x30f)
#define LWE397_SU_INST_15_SRC                       1:0
#define LWE397_SU_INST_15_SRC_VPE                 (0)
#define LWE397_SU_INST_15_SRC_Z                   (1)

#define LWE397_SU_INST_15_VC_ROW                    6:3

#define LWE397_SU_INST_15_TRAM_ROW                  14:9

#define LWE397_SU_INST_15_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_15_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_15_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_15_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_15_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_15_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_15_P0_POINT                  19:18
#define LWE397_SU_INST_15_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_15_P0_POINT_S                      (1)
#define LWE397_SU_INST_15_P0_POINT_T                      (2)

#define LWE397_SU_INST_15_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_15_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_15_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_15_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_15_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_15_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_15_P1_POINT                  23:22
#define LWE397_SU_INST_15_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_15_P1_POINT_S                      (1)
#define LWE397_SU_INST_15_P1_POINT_T                      (2)

#define LWE397_SU_INST_15_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_15_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_15_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_15_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_15_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_15_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_15_P2_POINT                  27:26
#define LWE397_SU_INST_15_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_15_P2_POINT_S                      (1)
#define LWE397_SU_INST_15_P2_POINT_T                      (2)

#define LWE397_SU_INST_15_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_15_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_15_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_15_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_15_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_15_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_15_P3_POINT                  31:30
#define LWE397_SU_INST_15_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_15_P3_POINT_S                      (1)
#define LWE397_SU_INST_15_P3_POINT_T                      (2)

#define LWE397_SU_INST_15_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_15_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_15_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_15_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_15_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_15_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_15_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_15_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_15_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_15_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_15_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_15_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_15_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_15_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_15_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_15_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_15_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_15_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_15_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_15_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_15_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_15_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_15_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_15_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_15_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_15_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_15_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_15_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_15_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_15_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_15_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_15_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_15_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_15_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_15_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_15_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_16  
#define LWE397_SU_INST_16                 (0x310)
#define LWE397_SU_INST_16_SRC                       1:0
#define LWE397_SU_INST_16_SRC_VPE                 (0)
#define LWE397_SU_INST_16_SRC_Z                   (1)

#define LWE397_SU_INST_16_VC_ROW                    6:3

#define LWE397_SU_INST_16_TRAM_ROW                  14:9

#define LWE397_SU_INST_16_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_16_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_16_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_16_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_16_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_16_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_16_P0_POINT                  19:18
#define LWE397_SU_INST_16_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_16_P0_POINT_S                      (1)
#define LWE397_SU_INST_16_P0_POINT_T                      (2)

#define LWE397_SU_INST_16_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_16_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_16_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_16_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_16_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_16_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_16_P1_POINT                  23:22
#define LWE397_SU_INST_16_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_16_P1_POINT_S                      (1)
#define LWE397_SU_INST_16_P1_POINT_T                      (2)

#define LWE397_SU_INST_16_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_16_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_16_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_16_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_16_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_16_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_16_P2_POINT                  27:26
#define LWE397_SU_INST_16_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_16_P2_POINT_S                      (1)
#define LWE397_SU_INST_16_P2_POINT_T                      (2)

#define LWE397_SU_INST_16_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_16_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_16_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_16_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_16_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_16_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_16_P3_POINT                  31:30
#define LWE397_SU_INST_16_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_16_P3_POINT_S                      (1)
#define LWE397_SU_INST_16_P3_POINT_T                      (2)

#define LWE397_SU_INST_16_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_16_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_16_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_16_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_16_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_16_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_16_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_16_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_16_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_16_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_16_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_16_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_16_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_16_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_16_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_16_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_16_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_16_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_16_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_16_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_16_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_16_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_16_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_16_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_16_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_16_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_16_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_16_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_16_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_16_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_16_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_16_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_16_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_16_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_16_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_16_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_17  
#define LWE397_SU_INST_17                 (0x311)
#define LWE397_SU_INST_17_SRC                       1:0
#define LWE397_SU_INST_17_SRC_VPE                 (0)
#define LWE397_SU_INST_17_SRC_Z                   (1)

#define LWE397_SU_INST_17_VC_ROW                    6:3

#define LWE397_SU_INST_17_TRAM_ROW                  14:9

#define LWE397_SU_INST_17_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_17_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_17_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_17_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_17_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_17_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_17_P0_POINT                  19:18
#define LWE397_SU_INST_17_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_17_P0_POINT_S                      (1)
#define LWE397_SU_INST_17_P0_POINT_T                      (2)

#define LWE397_SU_INST_17_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_17_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_17_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_17_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_17_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_17_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_17_P1_POINT                  23:22
#define LWE397_SU_INST_17_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_17_P1_POINT_S                      (1)
#define LWE397_SU_INST_17_P1_POINT_T                      (2)

#define LWE397_SU_INST_17_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_17_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_17_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_17_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_17_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_17_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_17_P2_POINT                  27:26
#define LWE397_SU_INST_17_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_17_P2_POINT_S                      (1)
#define LWE397_SU_INST_17_P2_POINT_T                      (2)

#define LWE397_SU_INST_17_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_17_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_17_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_17_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_17_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_17_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_17_P3_POINT                  31:30
#define LWE397_SU_INST_17_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_17_P3_POINT_S                      (1)
#define LWE397_SU_INST_17_P3_POINT_T                      (2)

#define LWE397_SU_INST_17_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_17_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_17_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_17_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_17_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_17_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_17_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_17_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_17_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_17_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_17_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_17_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_17_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_17_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_17_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_17_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_17_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_17_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_17_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_17_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_17_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_17_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_17_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_17_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_17_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_17_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_17_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_17_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_17_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_17_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_17_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_17_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_17_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_17_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_17_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_17_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_18  
#define LWE397_SU_INST_18                 (0x312)
#define LWE397_SU_INST_18_SRC                       1:0
#define LWE397_SU_INST_18_SRC_VPE                 (0)
#define LWE397_SU_INST_18_SRC_Z                   (1)

#define LWE397_SU_INST_18_VC_ROW                    6:3

#define LWE397_SU_INST_18_TRAM_ROW                  14:9

#define LWE397_SU_INST_18_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_18_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_18_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_18_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_18_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_18_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_18_P0_POINT                  19:18
#define LWE397_SU_INST_18_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_18_P0_POINT_S                      (1)
#define LWE397_SU_INST_18_P0_POINT_T                      (2)

#define LWE397_SU_INST_18_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_18_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_18_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_18_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_18_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_18_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_18_P1_POINT                  23:22
#define LWE397_SU_INST_18_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_18_P1_POINT_S                      (1)
#define LWE397_SU_INST_18_P1_POINT_T                      (2)

#define LWE397_SU_INST_18_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_18_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_18_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_18_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_18_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_18_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_18_P2_POINT                  27:26
#define LWE397_SU_INST_18_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_18_P2_POINT_S                      (1)
#define LWE397_SU_INST_18_P2_POINT_T                      (2)

#define LWE397_SU_INST_18_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_18_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_18_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_18_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_18_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_18_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_18_P3_POINT                  31:30
#define LWE397_SU_INST_18_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_18_P3_POINT_S                      (1)
#define LWE397_SU_INST_18_P3_POINT_T                      (2)

#define LWE397_SU_INST_18_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_18_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_18_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_18_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_18_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_18_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_18_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_18_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_18_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_18_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_18_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_18_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_18_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_18_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_18_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_18_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_18_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_18_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_18_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_18_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_18_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_18_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_18_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_18_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_18_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_18_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_18_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_18_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_18_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_18_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_18_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_18_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_18_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_18_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_18_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_18_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_19  
#define LWE397_SU_INST_19                 (0x313)
#define LWE397_SU_INST_19_SRC                       1:0
#define LWE397_SU_INST_19_SRC_VPE                 (0)
#define LWE397_SU_INST_19_SRC_Z                   (1)

#define LWE397_SU_INST_19_VC_ROW                    6:3

#define LWE397_SU_INST_19_TRAM_ROW                  14:9

#define LWE397_SU_INST_19_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_19_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_19_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_19_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_19_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_19_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_19_P0_POINT                  19:18
#define LWE397_SU_INST_19_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_19_P0_POINT_S                      (1)
#define LWE397_SU_INST_19_P0_POINT_T                      (2)

#define LWE397_SU_INST_19_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_19_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_19_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_19_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_19_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_19_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_19_P1_POINT                  23:22
#define LWE397_SU_INST_19_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_19_P1_POINT_S                      (1)
#define LWE397_SU_INST_19_P1_POINT_T                      (2)

#define LWE397_SU_INST_19_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_19_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_19_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_19_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_19_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_19_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_19_P2_POINT                  27:26
#define LWE397_SU_INST_19_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_19_P2_POINT_S                      (1)
#define LWE397_SU_INST_19_P2_POINT_T                      (2)

#define LWE397_SU_INST_19_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_19_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_19_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_19_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_19_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_19_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_19_P3_POINT                  31:30
#define LWE397_SU_INST_19_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_19_P3_POINT_S                      (1)
#define LWE397_SU_INST_19_P3_POINT_T                      (2)

#define LWE397_SU_INST_19_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_19_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_19_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_19_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_19_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_19_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_19_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_19_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_19_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_19_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_19_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_19_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_19_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_19_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_19_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_19_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_19_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_19_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_19_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_19_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_19_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_19_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_19_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_19_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_19_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_19_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_19_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_19_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_19_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_19_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_19_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_19_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_19_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_19_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_19_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_19_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_20  
#define LWE397_SU_INST_20                 (0x314)
#define LWE397_SU_INST_20_SRC                       1:0
#define LWE397_SU_INST_20_SRC_VPE                 (0)
#define LWE397_SU_INST_20_SRC_Z                   (1)

#define LWE397_SU_INST_20_VC_ROW                    6:3

#define LWE397_SU_INST_20_TRAM_ROW                  14:9

#define LWE397_SU_INST_20_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_20_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_20_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_20_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_20_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_20_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_20_P0_POINT                  19:18
#define LWE397_SU_INST_20_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_20_P0_POINT_S                      (1)
#define LWE397_SU_INST_20_P0_POINT_T                      (2)

#define LWE397_SU_INST_20_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_20_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_20_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_20_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_20_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_20_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_20_P1_POINT                  23:22
#define LWE397_SU_INST_20_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_20_P1_POINT_S                      (1)
#define LWE397_SU_INST_20_P1_POINT_T                      (2)

#define LWE397_SU_INST_20_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_20_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_20_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_20_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_20_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_20_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_20_P2_POINT                  27:26
#define LWE397_SU_INST_20_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_20_P2_POINT_S                      (1)
#define LWE397_SU_INST_20_P2_POINT_T                      (2)

#define LWE397_SU_INST_20_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_20_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_20_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_20_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_20_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_20_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_20_P3_POINT                  31:30
#define LWE397_SU_INST_20_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_20_P3_POINT_S                      (1)
#define LWE397_SU_INST_20_P3_POINT_T                      (2)

#define LWE397_SU_INST_20_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_20_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_20_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_20_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_20_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_20_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_20_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_20_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_20_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_20_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_20_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_20_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_20_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_20_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_20_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_20_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_20_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_20_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_20_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_20_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_20_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_20_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_20_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_20_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_20_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_20_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_20_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_20_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_20_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_20_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_20_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_20_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_20_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_20_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_20_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_20_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_21  
#define LWE397_SU_INST_21                 (0x315)
#define LWE397_SU_INST_21_SRC                       1:0
#define LWE397_SU_INST_21_SRC_VPE                 (0)
#define LWE397_SU_INST_21_SRC_Z                   (1)

#define LWE397_SU_INST_21_VC_ROW                    6:3

#define LWE397_SU_INST_21_TRAM_ROW                  14:9

#define LWE397_SU_INST_21_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_21_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_21_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_21_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_21_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_21_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_21_P0_POINT                  19:18
#define LWE397_SU_INST_21_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_21_P0_POINT_S                      (1)
#define LWE397_SU_INST_21_P0_POINT_T                      (2)

#define LWE397_SU_INST_21_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_21_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_21_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_21_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_21_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_21_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_21_P1_POINT                  23:22
#define LWE397_SU_INST_21_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_21_P1_POINT_S                      (1)
#define LWE397_SU_INST_21_P1_POINT_T                      (2)

#define LWE397_SU_INST_21_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_21_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_21_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_21_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_21_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_21_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_21_P2_POINT                  27:26
#define LWE397_SU_INST_21_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_21_P2_POINT_S                      (1)
#define LWE397_SU_INST_21_P2_POINT_T                      (2)

#define LWE397_SU_INST_21_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_21_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_21_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_21_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_21_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_21_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_21_P3_POINT                  31:30
#define LWE397_SU_INST_21_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_21_P3_POINT_S                      (1)
#define LWE397_SU_INST_21_P3_POINT_T                      (2)

#define LWE397_SU_INST_21_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_21_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_21_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_21_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_21_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_21_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_21_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_21_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_21_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_21_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_21_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_21_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_21_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_21_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_21_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_21_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_21_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_21_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_21_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_21_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_21_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_21_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_21_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_21_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_21_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_21_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_21_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_21_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_21_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_21_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_21_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_21_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_21_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_21_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_21_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_21_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_22  
#define LWE397_SU_INST_22                 (0x316)
#define LWE397_SU_INST_22_SRC                       1:0
#define LWE397_SU_INST_22_SRC_VPE                 (0)
#define LWE397_SU_INST_22_SRC_Z                   (1)

#define LWE397_SU_INST_22_VC_ROW                    6:3

#define LWE397_SU_INST_22_TRAM_ROW                  14:9

#define LWE397_SU_INST_22_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_22_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_22_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_22_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_22_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_22_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_22_P0_POINT                  19:18
#define LWE397_SU_INST_22_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_22_P0_POINT_S                      (1)
#define LWE397_SU_INST_22_P0_POINT_T                      (2)

#define LWE397_SU_INST_22_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_22_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_22_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_22_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_22_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_22_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_22_P1_POINT                  23:22
#define LWE397_SU_INST_22_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_22_P1_POINT_S                      (1)
#define LWE397_SU_INST_22_P1_POINT_T                      (2)

#define LWE397_SU_INST_22_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_22_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_22_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_22_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_22_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_22_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_22_P2_POINT                  27:26
#define LWE397_SU_INST_22_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_22_P2_POINT_S                      (1)
#define LWE397_SU_INST_22_P2_POINT_T                      (2)

#define LWE397_SU_INST_22_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_22_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_22_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_22_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_22_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_22_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_22_P3_POINT                  31:30
#define LWE397_SU_INST_22_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_22_P3_POINT_S                      (1)
#define LWE397_SU_INST_22_P3_POINT_T                      (2)

#define LWE397_SU_INST_22_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_22_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_22_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_22_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_22_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_22_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_22_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_22_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_22_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_22_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_22_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_22_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_22_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_22_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_22_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_22_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_22_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_22_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_22_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_22_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_22_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_22_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_22_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_22_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_22_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_22_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_22_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_22_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_22_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_22_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_22_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_22_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_22_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_22_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_22_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_22_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_23  
#define LWE397_SU_INST_23                 (0x317)
#define LWE397_SU_INST_23_SRC                       1:0
#define LWE397_SU_INST_23_SRC_VPE                 (0)
#define LWE397_SU_INST_23_SRC_Z                   (1)

#define LWE397_SU_INST_23_VC_ROW                    6:3

#define LWE397_SU_INST_23_TRAM_ROW                  14:9

#define LWE397_SU_INST_23_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_23_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_23_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_23_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_23_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_23_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_23_P0_POINT                  19:18
#define LWE397_SU_INST_23_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_23_P0_POINT_S                      (1)
#define LWE397_SU_INST_23_P0_POINT_T                      (2)

#define LWE397_SU_INST_23_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_23_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_23_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_23_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_23_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_23_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_23_P1_POINT                  23:22
#define LWE397_SU_INST_23_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_23_P1_POINT_S                      (1)
#define LWE397_SU_INST_23_P1_POINT_T                      (2)

#define LWE397_SU_INST_23_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_23_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_23_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_23_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_23_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_23_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_23_P2_POINT                  27:26
#define LWE397_SU_INST_23_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_23_P2_POINT_S                      (1)
#define LWE397_SU_INST_23_P2_POINT_T                      (2)

#define LWE397_SU_INST_23_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_23_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_23_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_23_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_23_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_23_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_23_P3_POINT                  31:30
#define LWE397_SU_INST_23_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_23_P3_POINT_S                      (1)
#define LWE397_SU_INST_23_P3_POINT_T                      (2)

#define LWE397_SU_INST_23_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_23_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_23_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_23_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_23_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_23_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_23_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_23_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_23_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_23_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_23_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_23_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_23_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_23_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_23_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_23_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_23_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_23_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_23_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_23_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_23_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_23_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_23_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_23_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_23_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_23_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_23_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_23_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_23_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_23_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_23_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_23_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_23_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_23_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_23_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_23_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_24  
#define LWE397_SU_INST_24                 (0x318)
#define LWE397_SU_INST_24_SRC                       1:0
#define LWE397_SU_INST_24_SRC_VPE                 (0)
#define LWE397_SU_INST_24_SRC_Z                   (1)

#define LWE397_SU_INST_24_VC_ROW                    6:3

#define LWE397_SU_INST_24_TRAM_ROW                  14:9

#define LWE397_SU_INST_24_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_24_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_24_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_24_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_24_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_24_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_24_P0_POINT                  19:18
#define LWE397_SU_INST_24_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_24_P0_POINT_S                      (1)
#define LWE397_SU_INST_24_P0_POINT_T                      (2)

#define LWE397_SU_INST_24_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_24_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_24_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_24_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_24_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_24_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_24_P1_POINT                  23:22
#define LWE397_SU_INST_24_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_24_P1_POINT_S                      (1)
#define LWE397_SU_INST_24_P1_POINT_T                      (2)

#define LWE397_SU_INST_24_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_24_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_24_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_24_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_24_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_24_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_24_P2_POINT                  27:26
#define LWE397_SU_INST_24_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_24_P2_POINT_S                      (1)
#define LWE397_SU_INST_24_P2_POINT_T                      (2)

#define LWE397_SU_INST_24_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_24_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_24_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_24_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_24_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_24_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_24_P3_POINT                  31:30
#define LWE397_SU_INST_24_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_24_P3_POINT_S                      (1)
#define LWE397_SU_INST_24_P3_POINT_T                      (2)

#define LWE397_SU_INST_24_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_24_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_24_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_24_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_24_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_24_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_24_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_24_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_24_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_24_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_24_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_24_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_24_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_24_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_24_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_24_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_24_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_24_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_24_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_24_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_24_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_24_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_24_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_24_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_24_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_24_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_24_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_24_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_24_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_24_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_24_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_24_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_24_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_24_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_24_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_24_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_25  
#define LWE397_SU_INST_25                 (0x319)
#define LWE397_SU_INST_25_SRC                       1:0
#define LWE397_SU_INST_25_SRC_VPE                 (0)
#define LWE397_SU_INST_25_SRC_Z                   (1)

#define LWE397_SU_INST_25_VC_ROW                    6:3

#define LWE397_SU_INST_25_TRAM_ROW                  14:9

#define LWE397_SU_INST_25_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_25_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_25_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_25_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_25_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_25_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_25_P0_POINT                  19:18
#define LWE397_SU_INST_25_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_25_P0_POINT_S                      (1)
#define LWE397_SU_INST_25_P0_POINT_T                      (2)

#define LWE397_SU_INST_25_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_25_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_25_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_25_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_25_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_25_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_25_P1_POINT                  23:22
#define LWE397_SU_INST_25_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_25_P1_POINT_S                      (1)
#define LWE397_SU_INST_25_P1_POINT_T                      (2)

#define LWE397_SU_INST_25_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_25_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_25_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_25_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_25_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_25_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_25_P2_POINT                  27:26
#define LWE397_SU_INST_25_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_25_P2_POINT_S                      (1)
#define LWE397_SU_INST_25_P2_POINT_T                      (2)

#define LWE397_SU_INST_25_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_25_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_25_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_25_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_25_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_25_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_25_P3_POINT                  31:30
#define LWE397_SU_INST_25_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_25_P3_POINT_S                      (1)
#define LWE397_SU_INST_25_P3_POINT_T                      (2)

#define LWE397_SU_INST_25_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_25_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_25_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_25_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_25_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_25_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_25_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_25_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_25_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_25_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_25_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_25_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_25_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_25_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_25_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_25_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_25_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_25_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_25_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_25_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_25_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_25_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_25_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_25_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_25_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_25_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_25_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_25_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_25_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_25_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_25_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_25_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_25_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_25_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_25_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_25_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_26  
#define LWE397_SU_INST_26                 (0x31a)
#define LWE397_SU_INST_26_SRC                       1:0
#define LWE397_SU_INST_26_SRC_VPE                 (0)
#define LWE397_SU_INST_26_SRC_Z                   (1)

#define LWE397_SU_INST_26_VC_ROW                    6:3

#define LWE397_SU_INST_26_TRAM_ROW                  14:9

#define LWE397_SU_INST_26_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_26_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_26_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_26_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_26_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_26_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_26_P0_POINT                  19:18
#define LWE397_SU_INST_26_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_26_P0_POINT_S                      (1)
#define LWE397_SU_INST_26_P0_POINT_T                      (2)

#define LWE397_SU_INST_26_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_26_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_26_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_26_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_26_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_26_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_26_P1_POINT                  23:22
#define LWE397_SU_INST_26_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_26_P1_POINT_S                      (1)
#define LWE397_SU_INST_26_P1_POINT_T                      (2)

#define LWE397_SU_INST_26_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_26_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_26_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_26_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_26_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_26_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_26_P2_POINT                  27:26
#define LWE397_SU_INST_26_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_26_P2_POINT_S                      (1)
#define LWE397_SU_INST_26_P2_POINT_T                      (2)

#define LWE397_SU_INST_26_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_26_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_26_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_26_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_26_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_26_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_26_P3_POINT                  31:30
#define LWE397_SU_INST_26_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_26_P3_POINT_S                      (1)
#define LWE397_SU_INST_26_P3_POINT_T                      (2)

#define LWE397_SU_INST_26_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_26_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_26_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_26_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_26_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_26_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_26_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_26_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_26_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_26_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_26_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_26_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_26_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_26_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_26_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_26_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_26_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_26_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_26_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_26_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_26_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_26_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_26_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_26_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_26_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_26_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_26_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_26_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_26_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_26_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_26_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_26_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_26_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_26_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_26_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_26_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_27  
#define LWE397_SU_INST_27                 (0x31b)
#define LWE397_SU_INST_27_SRC                       1:0
#define LWE397_SU_INST_27_SRC_VPE                 (0)
#define LWE397_SU_INST_27_SRC_Z                   (1)

#define LWE397_SU_INST_27_VC_ROW                    6:3

#define LWE397_SU_INST_27_TRAM_ROW                  14:9

#define LWE397_SU_INST_27_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_27_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_27_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_27_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_27_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_27_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_27_P0_POINT                  19:18
#define LWE397_SU_INST_27_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_27_P0_POINT_S                      (1)
#define LWE397_SU_INST_27_P0_POINT_T                      (2)

#define LWE397_SU_INST_27_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_27_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_27_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_27_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_27_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_27_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_27_P1_POINT                  23:22
#define LWE397_SU_INST_27_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_27_P1_POINT_S                      (1)
#define LWE397_SU_INST_27_P1_POINT_T                      (2)

#define LWE397_SU_INST_27_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_27_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_27_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_27_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_27_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_27_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_27_P2_POINT                  27:26
#define LWE397_SU_INST_27_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_27_P2_POINT_S                      (1)
#define LWE397_SU_INST_27_P2_POINT_T                      (2)

#define LWE397_SU_INST_27_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_27_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_27_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_27_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_27_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_27_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_27_P3_POINT                  31:30
#define LWE397_SU_INST_27_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_27_P3_POINT_S                      (1)
#define LWE397_SU_INST_27_P3_POINT_T                      (2)

#define LWE397_SU_INST_27_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_27_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_27_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_27_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_27_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_27_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_27_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_27_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_27_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_27_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_27_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_27_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_27_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_27_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_27_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_27_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_27_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_27_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_27_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_27_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_27_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_27_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_27_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_27_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_27_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_27_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_27_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_27_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_27_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_27_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_27_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_27_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_27_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_27_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_27_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_27_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_28  
#define LWE397_SU_INST_28                 (0x31c)
#define LWE397_SU_INST_28_SRC                       1:0
#define LWE397_SU_INST_28_SRC_VPE                 (0)
#define LWE397_SU_INST_28_SRC_Z                   (1)

#define LWE397_SU_INST_28_VC_ROW                    6:3

#define LWE397_SU_INST_28_TRAM_ROW                  14:9

#define LWE397_SU_INST_28_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_28_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_28_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_28_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_28_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_28_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_28_P0_POINT                  19:18
#define LWE397_SU_INST_28_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_28_P0_POINT_S                      (1)
#define LWE397_SU_INST_28_P0_POINT_T                      (2)

#define LWE397_SU_INST_28_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_28_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_28_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_28_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_28_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_28_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_28_P1_POINT                  23:22
#define LWE397_SU_INST_28_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_28_P1_POINT_S                      (1)
#define LWE397_SU_INST_28_P1_POINT_T                      (2)

#define LWE397_SU_INST_28_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_28_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_28_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_28_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_28_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_28_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_28_P2_POINT                  27:26
#define LWE397_SU_INST_28_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_28_P2_POINT_S                      (1)
#define LWE397_SU_INST_28_P2_POINT_T                      (2)

#define LWE397_SU_INST_28_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_28_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_28_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_28_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_28_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_28_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_28_P3_POINT                  31:30
#define LWE397_SU_INST_28_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_28_P3_POINT_S                      (1)
#define LWE397_SU_INST_28_P3_POINT_T                      (2)

#define LWE397_SU_INST_28_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_28_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_28_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_28_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_28_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_28_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_28_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_28_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_28_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_28_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_28_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_28_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_28_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_28_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_28_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_28_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_28_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_28_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_28_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_28_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_28_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_28_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_28_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_28_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_28_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_28_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_28_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_28_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_28_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_28_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_28_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_28_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_28_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_28_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_28_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_28_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_29  
#define LWE397_SU_INST_29                 (0x31d)
#define LWE397_SU_INST_29_SRC                       1:0
#define LWE397_SU_INST_29_SRC_VPE                 (0)
#define LWE397_SU_INST_29_SRC_Z                   (1)

#define LWE397_SU_INST_29_VC_ROW                    6:3

#define LWE397_SU_INST_29_TRAM_ROW                  14:9

#define LWE397_SU_INST_29_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_29_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_29_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_29_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_29_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_29_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_29_P0_POINT                  19:18
#define LWE397_SU_INST_29_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_29_P0_POINT_S                      (1)
#define LWE397_SU_INST_29_P0_POINT_T                      (2)

#define LWE397_SU_INST_29_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_29_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_29_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_29_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_29_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_29_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_29_P1_POINT                  23:22
#define LWE397_SU_INST_29_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_29_P1_POINT_S                      (1)
#define LWE397_SU_INST_29_P1_POINT_T                      (2)

#define LWE397_SU_INST_29_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_29_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_29_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_29_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_29_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_29_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_29_P2_POINT                  27:26
#define LWE397_SU_INST_29_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_29_P2_POINT_S                      (1)
#define LWE397_SU_INST_29_P2_POINT_T                      (2)

#define LWE397_SU_INST_29_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_29_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_29_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_29_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_29_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_29_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_29_P3_POINT                  31:30
#define LWE397_SU_INST_29_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_29_P3_POINT_S                      (1)
#define LWE397_SU_INST_29_P3_POINT_T                      (2)

#define LWE397_SU_INST_29_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_29_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_29_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_29_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_29_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_29_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_29_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_29_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_29_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_29_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_29_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_29_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_29_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_29_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_29_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_29_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_29_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_29_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_29_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_29_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_29_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_29_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_29_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_29_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_29_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_29_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_29_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_29_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_29_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_29_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_29_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_29_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_29_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_29_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_29_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_29_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_30  
#define LWE397_SU_INST_30                 (0x31e)
#define LWE397_SU_INST_30_SRC                       1:0
#define LWE397_SU_INST_30_SRC_VPE                 (0)
#define LWE397_SU_INST_30_SRC_Z                   (1)

#define LWE397_SU_INST_30_VC_ROW                    6:3

#define LWE397_SU_INST_30_TRAM_ROW                  14:9

#define LWE397_SU_INST_30_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_30_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_30_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_30_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_30_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_30_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_30_P0_POINT                  19:18
#define LWE397_SU_INST_30_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_30_P0_POINT_S                      (1)
#define LWE397_SU_INST_30_P0_POINT_T                      (2)

#define LWE397_SU_INST_30_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_30_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_30_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_30_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_30_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_30_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_30_P1_POINT                  23:22
#define LWE397_SU_INST_30_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_30_P1_POINT_S                      (1)
#define LWE397_SU_INST_30_P1_POINT_T                      (2)

#define LWE397_SU_INST_30_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_30_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_30_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_30_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_30_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_30_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_30_P2_POINT                  27:26
#define LWE397_SU_INST_30_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_30_P2_POINT_S                      (1)
#define LWE397_SU_INST_30_P2_POINT_T                      (2)

#define LWE397_SU_INST_30_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_30_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_30_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_30_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_30_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_30_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_30_P3_POINT                  31:30
#define LWE397_SU_INST_30_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_30_P3_POINT_S                      (1)
#define LWE397_SU_INST_30_P3_POINT_T                      (2)

#define LWE397_SU_INST_30_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_30_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_30_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_30_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_30_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_30_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_30_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_30_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_30_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_30_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_30_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_30_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_30_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_30_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_30_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_30_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_30_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_30_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_30_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_30_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_30_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_30_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_30_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_30_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_30_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_30_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_30_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_30_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_30_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_30_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_30_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_30_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_30_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_30_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_30_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_30_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_31  
#define LWE397_SU_INST_31                 (0x31f)
#define LWE397_SU_INST_31_SRC                       1:0
#define LWE397_SU_INST_31_SRC_VPE                 (0)
#define LWE397_SU_INST_31_SRC_Z                   (1)

#define LWE397_SU_INST_31_VC_ROW                    6:3

#define LWE397_SU_INST_31_TRAM_ROW                  14:9

#define LWE397_SU_INST_31_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_31_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_31_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_31_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_31_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_31_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_31_P0_POINT                  19:18
#define LWE397_SU_INST_31_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_31_P0_POINT_S                      (1)
#define LWE397_SU_INST_31_P0_POINT_T                      (2)

#define LWE397_SU_INST_31_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_31_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_31_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_31_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_31_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_31_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_31_P1_POINT                  23:22
#define LWE397_SU_INST_31_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_31_P1_POINT_S                      (1)
#define LWE397_SU_INST_31_P1_POINT_T                      (2)

#define LWE397_SU_INST_31_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_31_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_31_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_31_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_31_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_31_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_31_P2_POINT                  27:26
#define LWE397_SU_INST_31_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_31_P2_POINT_S                      (1)
#define LWE397_SU_INST_31_P2_POINT_T                      (2)

#define LWE397_SU_INST_31_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_31_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_31_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_31_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_31_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_31_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_31_P3_POINT                  31:30
#define LWE397_SU_INST_31_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_31_P3_POINT_S                      (1)
#define LWE397_SU_INST_31_P3_POINT_T                      (2)

#define LWE397_SU_INST_31_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_31_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_31_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_31_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_31_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_31_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_31_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_31_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_31_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_31_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_31_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_31_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_31_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_31_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_31_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_31_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_31_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_31_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_31_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_31_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_31_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_31_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_31_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_31_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_31_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_31_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_31_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_31_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_31_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_31_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_31_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_31_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_31_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_31_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_31_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_31_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_32  
#define LWE397_SU_INST_32                 (0x320)
#define LWE397_SU_INST_32_SRC                       1:0
#define LWE397_SU_INST_32_SRC_VPE                 (0)
#define LWE397_SU_INST_32_SRC_Z                   (1)

#define LWE397_SU_INST_32_VC_ROW                    6:3

#define LWE397_SU_INST_32_TRAM_ROW                  14:9

#define LWE397_SU_INST_32_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_32_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_32_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_32_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_32_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_32_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_32_P0_POINT                  19:18
#define LWE397_SU_INST_32_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_32_P0_POINT_S                      (1)
#define LWE397_SU_INST_32_P0_POINT_T                      (2)

#define LWE397_SU_INST_32_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_32_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_32_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_32_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_32_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_32_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_32_P1_POINT                  23:22
#define LWE397_SU_INST_32_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_32_P1_POINT_S                      (1)
#define LWE397_SU_INST_32_P1_POINT_T                      (2)

#define LWE397_SU_INST_32_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_32_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_32_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_32_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_32_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_32_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_32_P2_POINT                  27:26
#define LWE397_SU_INST_32_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_32_P2_POINT_S                      (1)
#define LWE397_SU_INST_32_P2_POINT_T                      (2)

#define LWE397_SU_INST_32_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_32_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_32_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_32_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_32_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_32_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_32_P3_POINT                  31:30
#define LWE397_SU_INST_32_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_32_P3_POINT_S                      (1)
#define LWE397_SU_INST_32_P3_POINT_T                      (2)

#define LWE397_SU_INST_32_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_32_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_32_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_32_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_32_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_32_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_32_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_32_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_32_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_32_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_32_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_32_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_32_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_32_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_32_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_32_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_32_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_32_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_32_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_32_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_32_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_32_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_32_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_32_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_32_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_32_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_32_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_32_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_32_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_32_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_32_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_32_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_32_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_32_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_32_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_32_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_33  
#define LWE397_SU_INST_33                 (0x321)
#define LWE397_SU_INST_33_SRC                       1:0
#define LWE397_SU_INST_33_SRC_VPE                 (0)
#define LWE397_SU_INST_33_SRC_Z                   (1)

#define LWE397_SU_INST_33_VC_ROW                    6:3

#define LWE397_SU_INST_33_TRAM_ROW                  14:9

#define LWE397_SU_INST_33_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_33_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_33_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_33_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_33_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_33_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_33_P0_POINT                  19:18
#define LWE397_SU_INST_33_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_33_P0_POINT_S                      (1)
#define LWE397_SU_INST_33_P0_POINT_T                      (2)

#define LWE397_SU_INST_33_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_33_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_33_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_33_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_33_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_33_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_33_P1_POINT                  23:22
#define LWE397_SU_INST_33_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_33_P1_POINT_S                      (1)
#define LWE397_SU_INST_33_P1_POINT_T                      (2)

#define LWE397_SU_INST_33_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_33_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_33_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_33_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_33_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_33_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_33_P2_POINT                  27:26
#define LWE397_SU_INST_33_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_33_P2_POINT_S                      (1)
#define LWE397_SU_INST_33_P2_POINT_T                      (2)

#define LWE397_SU_INST_33_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_33_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_33_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_33_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_33_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_33_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_33_P3_POINT                  31:30
#define LWE397_SU_INST_33_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_33_P3_POINT_S                      (1)
#define LWE397_SU_INST_33_P3_POINT_T                      (2)

#define LWE397_SU_INST_33_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_33_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_33_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_33_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_33_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_33_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_33_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_33_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_33_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_33_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_33_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_33_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_33_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_33_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_33_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_33_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_33_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_33_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_33_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_33_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_33_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_33_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_33_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_33_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_33_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_33_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_33_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_33_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_33_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_33_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_33_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_33_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_33_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_33_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_33_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_33_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_34  
#define LWE397_SU_INST_34                 (0x322)
#define LWE397_SU_INST_34_SRC                       1:0
#define LWE397_SU_INST_34_SRC_VPE                 (0)
#define LWE397_SU_INST_34_SRC_Z                   (1)

#define LWE397_SU_INST_34_VC_ROW                    6:3

#define LWE397_SU_INST_34_TRAM_ROW                  14:9

#define LWE397_SU_INST_34_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_34_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_34_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_34_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_34_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_34_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_34_P0_POINT                  19:18
#define LWE397_SU_INST_34_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_34_P0_POINT_S                      (1)
#define LWE397_SU_INST_34_P0_POINT_T                      (2)

#define LWE397_SU_INST_34_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_34_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_34_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_34_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_34_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_34_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_34_P1_POINT                  23:22
#define LWE397_SU_INST_34_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_34_P1_POINT_S                      (1)
#define LWE397_SU_INST_34_P1_POINT_T                      (2)

#define LWE397_SU_INST_34_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_34_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_34_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_34_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_34_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_34_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_34_P2_POINT                  27:26
#define LWE397_SU_INST_34_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_34_P2_POINT_S                      (1)
#define LWE397_SU_INST_34_P2_POINT_T                      (2)

#define LWE397_SU_INST_34_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_34_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_34_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_34_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_34_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_34_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_34_P3_POINT                  31:30
#define LWE397_SU_INST_34_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_34_P3_POINT_S                      (1)
#define LWE397_SU_INST_34_P3_POINT_T                      (2)

#define LWE397_SU_INST_34_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_34_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_34_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_34_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_34_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_34_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_34_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_34_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_34_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_34_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_34_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_34_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_34_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_34_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_34_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_34_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_34_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_34_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_34_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_34_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_34_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_34_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_34_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_34_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_34_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_34_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_34_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_34_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_34_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_34_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_34_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_34_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_34_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_34_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_34_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_34_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_35  
#define LWE397_SU_INST_35                 (0x323)
#define LWE397_SU_INST_35_SRC                       1:0
#define LWE397_SU_INST_35_SRC_VPE                 (0)
#define LWE397_SU_INST_35_SRC_Z                   (1)

#define LWE397_SU_INST_35_VC_ROW                    6:3

#define LWE397_SU_INST_35_TRAM_ROW                  14:9

#define LWE397_SU_INST_35_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_35_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_35_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_35_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_35_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_35_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_35_P0_POINT                  19:18
#define LWE397_SU_INST_35_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_35_P0_POINT_S                      (1)
#define LWE397_SU_INST_35_P0_POINT_T                      (2)

#define LWE397_SU_INST_35_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_35_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_35_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_35_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_35_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_35_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_35_P1_POINT                  23:22
#define LWE397_SU_INST_35_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_35_P1_POINT_S                      (1)
#define LWE397_SU_INST_35_P1_POINT_T                      (2)

#define LWE397_SU_INST_35_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_35_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_35_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_35_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_35_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_35_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_35_P2_POINT                  27:26
#define LWE397_SU_INST_35_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_35_P2_POINT_S                      (1)
#define LWE397_SU_INST_35_P2_POINT_T                      (2)

#define LWE397_SU_INST_35_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_35_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_35_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_35_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_35_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_35_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_35_P3_POINT                  31:30
#define LWE397_SU_INST_35_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_35_P3_POINT_S                      (1)
#define LWE397_SU_INST_35_P3_POINT_T                      (2)

#define LWE397_SU_INST_35_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_35_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_35_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_35_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_35_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_35_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_35_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_35_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_35_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_35_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_35_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_35_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_35_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_35_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_35_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_35_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_35_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_35_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_35_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_35_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_35_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_35_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_35_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_35_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_35_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_35_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_35_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_35_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_35_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_35_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_35_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_35_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_35_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_35_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_35_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_35_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_36  
#define LWE397_SU_INST_36                 (0x324)
#define LWE397_SU_INST_36_SRC                       1:0
#define LWE397_SU_INST_36_SRC_VPE                 (0)
#define LWE397_SU_INST_36_SRC_Z                   (1)

#define LWE397_SU_INST_36_VC_ROW                    6:3

#define LWE397_SU_INST_36_TRAM_ROW                  14:9

#define LWE397_SU_INST_36_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_36_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_36_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_36_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_36_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_36_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_36_P0_POINT                  19:18
#define LWE397_SU_INST_36_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_36_P0_POINT_S                      (1)
#define LWE397_SU_INST_36_P0_POINT_T                      (2)

#define LWE397_SU_INST_36_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_36_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_36_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_36_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_36_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_36_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_36_P1_POINT                  23:22
#define LWE397_SU_INST_36_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_36_P1_POINT_S                      (1)
#define LWE397_SU_INST_36_P1_POINT_T                      (2)

#define LWE397_SU_INST_36_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_36_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_36_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_36_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_36_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_36_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_36_P2_POINT                  27:26
#define LWE397_SU_INST_36_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_36_P2_POINT_S                      (1)
#define LWE397_SU_INST_36_P2_POINT_T                      (2)

#define LWE397_SU_INST_36_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_36_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_36_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_36_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_36_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_36_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_36_P3_POINT                  31:30
#define LWE397_SU_INST_36_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_36_P3_POINT_S                      (1)
#define LWE397_SU_INST_36_P3_POINT_T                      (2)

#define LWE397_SU_INST_36_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_36_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_36_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_36_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_36_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_36_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_36_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_36_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_36_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_36_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_36_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_36_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_36_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_36_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_36_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_36_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_36_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_36_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_36_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_36_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_36_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_36_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_36_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_36_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_36_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_36_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_36_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_36_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_36_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_36_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_36_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_36_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_36_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_36_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_36_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_36_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_37  
#define LWE397_SU_INST_37                 (0x325)
#define LWE397_SU_INST_37_SRC                       1:0
#define LWE397_SU_INST_37_SRC_VPE                 (0)
#define LWE397_SU_INST_37_SRC_Z                   (1)

#define LWE397_SU_INST_37_VC_ROW                    6:3

#define LWE397_SU_INST_37_TRAM_ROW                  14:9

#define LWE397_SU_INST_37_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_37_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_37_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_37_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_37_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_37_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_37_P0_POINT                  19:18
#define LWE397_SU_INST_37_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_37_P0_POINT_S                      (1)
#define LWE397_SU_INST_37_P0_POINT_T                      (2)

#define LWE397_SU_INST_37_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_37_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_37_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_37_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_37_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_37_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_37_P1_POINT                  23:22
#define LWE397_SU_INST_37_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_37_P1_POINT_S                      (1)
#define LWE397_SU_INST_37_P1_POINT_T                      (2)

#define LWE397_SU_INST_37_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_37_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_37_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_37_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_37_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_37_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_37_P2_POINT                  27:26
#define LWE397_SU_INST_37_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_37_P2_POINT_S                      (1)
#define LWE397_SU_INST_37_P2_POINT_T                      (2)

#define LWE397_SU_INST_37_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_37_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_37_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_37_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_37_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_37_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_37_P3_POINT                  31:30
#define LWE397_SU_INST_37_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_37_P3_POINT_S                      (1)
#define LWE397_SU_INST_37_P3_POINT_T                      (2)

#define LWE397_SU_INST_37_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_37_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_37_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_37_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_37_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_37_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_37_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_37_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_37_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_37_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_37_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_37_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_37_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_37_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_37_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_37_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_37_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_37_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_37_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_37_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_37_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_37_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_37_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_37_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_37_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_37_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_37_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_37_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_37_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_37_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_37_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_37_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_37_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_37_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_37_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_37_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_38  
#define LWE397_SU_INST_38                 (0x326)
#define LWE397_SU_INST_38_SRC                       1:0
#define LWE397_SU_INST_38_SRC_VPE                 (0)
#define LWE397_SU_INST_38_SRC_Z                   (1)

#define LWE397_SU_INST_38_VC_ROW                    6:3

#define LWE397_SU_INST_38_TRAM_ROW                  14:9

#define LWE397_SU_INST_38_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_38_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_38_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_38_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_38_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_38_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_38_P0_POINT                  19:18
#define LWE397_SU_INST_38_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_38_P0_POINT_S                      (1)
#define LWE397_SU_INST_38_P0_POINT_T                      (2)

#define LWE397_SU_INST_38_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_38_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_38_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_38_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_38_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_38_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_38_P1_POINT                  23:22
#define LWE397_SU_INST_38_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_38_P1_POINT_S                      (1)
#define LWE397_SU_INST_38_P1_POINT_T                      (2)

#define LWE397_SU_INST_38_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_38_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_38_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_38_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_38_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_38_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_38_P2_POINT                  27:26
#define LWE397_SU_INST_38_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_38_P2_POINT_S                      (1)
#define LWE397_SU_INST_38_P2_POINT_T                      (2)

#define LWE397_SU_INST_38_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_38_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_38_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_38_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_38_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_38_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_38_P3_POINT                  31:30
#define LWE397_SU_INST_38_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_38_P3_POINT_S                      (1)
#define LWE397_SU_INST_38_P3_POINT_T                      (2)

#define LWE397_SU_INST_38_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_38_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_38_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_38_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_38_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_38_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_38_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_38_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_38_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_38_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_38_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_38_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_38_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_38_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_38_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_38_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_38_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_38_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_38_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_38_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_38_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_38_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_38_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_38_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_38_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_38_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_38_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_38_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_38_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_38_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_38_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_38_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_38_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_38_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_38_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_38_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_39  
#define LWE397_SU_INST_39                 (0x327)
#define LWE397_SU_INST_39_SRC                       1:0
#define LWE397_SU_INST_39_SRC_VPE                 (0)
#define LWE397_SU_INST_39_SRC_Z                   (1)

#define LWE397_SU_INST_39_VC_ROW                    6:3

#define LWE397_SU_INST_39_TRAM_ROW                  14:9

#define LWE397_SU_INST_39_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_39_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_39_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_39_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_39_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_39_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_39_P0_POINT                  19:18
#define LWE397_SU_INST_39_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_39_P0_POINT_S                      (1)
#define LWE397_SU_INST_39_P0_POINT_T                      (2)

#define LWE397_SU_INST_39_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_39_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_39_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_39_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_39_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_39_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_39_P1_POINT                  23:22
#define LWE397_SU_INST_39_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_39_P1_POINT_S                      (1)
#define LWE397_SU_INST_39_P1_POINT_T                      (2)

#define LWE397_SU_INST_39_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_39_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_39_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_39_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_39_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_39_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_39_P2_POINT                  27:26
#define LWE397_SU_INST_39_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_39_P2_POINT_S                      (1)
#define LWE397_SU_INST_39_P2_POINT_T                      (2)

#define LWE397_SU_INST_39_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_39_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_39_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_39_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_39_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_39_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_39_P3_POINT                  31:30
#define LWE397_SU_INST_39_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_39_P3_POINT_S                      (1)
#define LWE397_SU_INST_39_P3_POINT_T                      (2)

#define LWE397_SU_INST_39_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_39_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_39_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_39_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_39_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_39_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_39_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_39_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_39_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_39_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_39_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_39_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_39_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_39_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_39_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_39_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_39_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_39_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_39_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_39_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_39_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_39_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_39_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_39_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_39_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_39_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_39_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_39_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_39_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_39_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_39_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_39_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_39_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_39_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_39_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_39_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_40  
#define LWE397_SU_INST_40                 (0x328)
#define LWE397_SU_INST_40_SRC                       1:0
#define LWE397_SU_INST_40_SRC_VPE                 (0)
#define LWE397_SU_INST_40_SRC_Z                   (1)

#define LWE397_SU_INST_40_VC_ROW                    6:3

#define LWE397_SU_INST_40_TRAM_ROW                  14:9

#define LWE397_SU_INST_40_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_40_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_40_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_40_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_40_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_40_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_40_P0_POINT                  19:18
#define LWE397_SU_INST_40_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_40_P0_POINT_S                      (1)
#define LWE397_SU_INST_40_P0_POINT_T                      (2)

#define LWE397_SU_INST_40_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_40_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_40_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_40_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_40_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_40_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_40_P1_POINT                  23:22
#define LWE397_SU_INST_40_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_40_P1_POINT_S                      (1)
#define LWE397_SU_INST_40_P1_POINT_T                      (2)

#define LWE397_SU_INST_40_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_40_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_40_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_40_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_40_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_40_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_40_P2_POINT                  27:26
#define LWE397_SU_INST_40_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_40_P2_POINT_S                      (1)
#define LWE397_SU_INST_40_P2_POINT_T                      (2)

#define LWE397_SU_INST_40_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_40_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_40_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_40_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_40_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_40_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_40_P3_POINT                  31:30
#define LWE397_SU_INST_40_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_40_P3_POINT_S                      (1)
#define LWE397_SU_INST_40_P3_POINT_T                      (2)

#define LWE397_SU_INST_40_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_40_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_40_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_40_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_40_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_40_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_40_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_40_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_40_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_40_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_40_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_40_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_40_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_40_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_40_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_40_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_40_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_40_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_40_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_40_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_40_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_40_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_40_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_40_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_40_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_40_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_40_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_40_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_40_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_40_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_40_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_40_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_40_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_40_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_40_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_40_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_41  
#define LWE397_SU_INST_41                 (0x329)
#define LWE397_SU_INST_41_SRC                       1:0
#define LWE397_SU_INST_41_SRC_VPE                 (0)
#define LWE397_SU_INST_41_SRC_Z                   (1)

#define LWE397_SU_INST_41_VC_ROW                    6:3

#define LWE397_SU_INST_41_TRAM_ROW                  14:9

#define LWE397_SU_INST_41_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_41_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_41_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_41_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_41_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_41_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_41_P0_POINT                  19:18
#define LWE397_SU_INST_41_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_41_P0_POINT_S                      (1)
#define LWE397_SU_INST_41_P0_POINT_T                      (2)

#define LWE397_SU_INST_41_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_41_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_41_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_41_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_41_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_41_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_41_P1_POINT                  23:22
#define LWE397_SU_INST_41_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_41_P1_POINT_S                      (1)
#define LWE397_SU_INST_41_P1_POINT_T                      (2)

#define LWE397_SU_INST_41_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_41_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_41_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_41_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_41_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_41_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_41_P2_POINT                  27:26
#define LWE397_SU_INST_41_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_41_P2_POINT_S                      (1)
#define LWE397_SU_INST_41_P2_POINT_T                      (2)

#define LWE397_SU_INST_41_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_41_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_41_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_41_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_41_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_41_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_41_P3_POINT                  31:30
#define LWE397_SU_INST_41_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_41_P3_POINT_S                      (1)
#define LWE397_SU_INST_41_P3_POINT_T                      (2)

#define LWE397_SU_INST_41_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_41_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_41_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_41_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_41_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_41_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_41_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_41_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_41_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_41_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_41_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_41_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_41_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_41_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_41_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_41_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_41_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_41_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_41_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_41_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_41_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_41_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_41_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_41_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_41_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_41_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_41_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_41_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_41_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_41_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_41_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_41_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_41_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_41_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_41_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_41_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_42  
#define LWE397_SU_INST_42                 (0x32a)
#define LWE397_SU_INST_42_SRC                       1:0
#define LWE397_SU_INST_42_SRC_VPE                 (0)
#define LWE397_SU_INST_42_SRC_Z                   (1)

#define LWE397_SU_INST_42_VC_ROW                    6:3

#define LWE397_SU_INST_42_TRAM_ROW                  14:9

#define LWE397_SU_INST_42_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_42_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_42_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_42_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_42_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_42_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_42_P0_POINT                  19:18
#define LWE397_SU_INST_42_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_42_P0_POINT_S                      (1)
#define LWE397_SU_INST_42_P0_POINT_T                      (2)

#define LWE397_SU_INST_42_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_42_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_42_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_42_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_42_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_42_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_42_P1_POINT                  23:22
#define LWE397_SU_INST_42_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_42_P1_POINT_S                      (1)
#define LWE397_SU_INST_42_P1_POINT_T                      (2)

#define LWE397_SU_INST_42_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_42_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_42_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_42_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_42_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_42_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_42_P2_POINT                  27:26
#define LWE397_SU_INST_42_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_42_P2_POINT_S                      (1)
#define LWE397_SU_INST_42_P2_POINT_T                      (2)

#define LWE397_SU_INST_42_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_42_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_42_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_42_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_42_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_42_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_42_P3_POINT                  31:30
#define LWE397_SU_INST_42_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_42_P3_POINT_S                      (1)
#define LWE397_SU_INST_42_P3_POINT_T                      (2)

#define LWE397_SU_INST_42_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_42_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_42_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_42_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_42_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_42_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_42_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_42_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_42_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_42_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_42_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_42_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_42_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_42_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_42_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_42_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_42_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_42_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_42_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_42_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_42_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_42_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_42_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_42_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_42_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_42_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_42_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_42_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_42_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_42_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_42_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_42_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_42_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_42_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_42_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_42_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_43  
#define LWE397_SU_INST_43                 (0x32b)
#define LWE397_SU_INST_43_SRC                       1:0
#define LWE397_SU_INST_43_SRC_VPE                 (0)
#define LWE397_SU_INST_43_SRC_Z                   (1)

#define LWE397_SU_INST_43_VC_ROW                    6:3

#define LWE397_SU_INST_43_TRAM_ROW                  14:9

#define LWE397_SU_INST_43_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_43_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_43_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_43_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_43_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_43_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_43_P0_POINT                  19:18
#define LWE397_SU_INST_43_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_43_P0_POINT_S                      (1)
#define LWE397_SU_INST_43_P0_POINT_T                      (2)

#define LWE397_SU_INST_43_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_43_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_43_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_43_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_43_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_43_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_43_P1_POINT                  23:22
#define LWE397_SU_INST_43_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_43_P1_POINT_S                      (1)
#define LWE397_SU_INST_43_P1_POINT_T                      (2)

#define LWE397_SU_INST_43_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_43_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_43_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_43_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_43_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_43_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_43_P2_POINT                  27:26
#define LWE397_SU_INST_43_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_43_P2_POINT_S                      (1)
#define LWE397_SU_INST_43_P2_POINT_T                      (2)

#define LWE397_SU_INST_43_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_43_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_43_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_43_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_43_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_43_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_43_P3_POINT                  31:30
#define LWE397_SU_INST_43_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_43_P3_POINT_S                      (1)
#define LWE397_SU_INST_43_P3_POINT_T                      (2)

#define LWE397_SU_INST_43_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_43_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_43_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_43_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_43_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_43_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_43_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_43_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_43_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_43_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_43_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_43_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_43_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_43_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_43_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_43_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_43_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_43_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_43_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_43_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_43_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_43_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_43_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_43_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_43_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_43_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_43_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_43_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_43_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_43_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_43_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_43_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_43_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_43_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_43_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_43_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_44  
#define LWE397_SU_INST_44                 (0x32c)
#define LWE397_SU_INST_44_SRC                       1:0
#define LWE397_SU_INST_44_SRC_VPE                 (0)
#define LWE397_SU_INST_44_SRC_Z                   (1)

#define LWE397_SU_INST_44_VC_ROW                    6:3

#define LWE397_SU_INST_44_TRAM_ROW                  14:9

#define LWE397_SU_INST_44_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_44_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_44_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_44_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_44_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_44_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_44_P0_POINT                  19:18
#define LWE397_SU_INST_44_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_44_P0_POINT_S                      (1)
#define LWE397_SU_INST_44_P0_POINT_T                      (2)

#define LWE397_SU_INST_44_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_44_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_44_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_44_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_44_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_44_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_44_P1_POINT                  23:22
#define LWE397_SU_INST_44_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_44_P1_POINT_S                      (1)
#define LWE397_SU_INST_44_P1_POINT_T                      (2)

#define LWE397_SU_INST_44_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_44_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_44_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_44_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_44_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_44_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_44_P2_POINT                  27:26
#define LWE397_SU_INST_44_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_44_P2_POINT_S                      (1)
#define LWE397_SU_INST_44_P2_POINT_T                      (2)

#define LWE397_SU_INST_44_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_44_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_44_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_44_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_44_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_44_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_44_P3_POINT                  31:30
#define LWE397_SU_INST_44_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_44_P3_POINT_S                      (1)
#define LWE397_SU_INST_44_P3_POINT_T                      (2)

#define LWE397_SU_INST_44_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_44_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_44_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_44_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_44_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_44_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_44_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_44_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_44_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_44_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_44_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_44_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_44_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_44_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_44_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_44_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_44_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_44_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_44_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_44_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_44_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_44_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_44_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_44_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_44_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_44_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_44_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_44_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_44_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_44_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_44_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_44_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_44_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_44_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_44_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_44_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_45  
#define LWE397_SU_INST_45                 (0x32d)
#define LWE397_SU_INST_45_SRC                       1:0
#define LWE397_SU_INST_45_SRC_VPE                 (0)
#define LWE397_SU_INST_45_SRC_Z                   (1)

#define LWE397_SU_INST_45_VC_ROW                    6:3

#define LWE397_SU_INST_45_TRAM_ROW                  14:9

#define LWE397_SU_INST_45_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_45_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_45_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_45_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_45_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_45_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_45_P0_POINT                  19:18
#define LWE397_SU_INST_45_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_45_P0_POINT_S                      (1)
#define LWE397_SU_INST_45_P0_POINT_T                      (2)

#define LWE397_SU_INST_45_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_45_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_45_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_45_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_45_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_45_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_45_P1_POINT                  23:22
#define LWE397_SU_INST_45_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_45_P1_POINT_S                      (1)
#define LWE397_SU_INST_45_P1_POINT_T                      (2)

#define LWE397_SU_INST_45_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_45_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_45_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_45_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_45_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_45_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_45_P2_POINT                  27:26
#define LWE397_SU_INST_45_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_45_P2_POINT_S                      (1)
#define LWE397_SU_INST_45_P2_POINT_T                      (2)

#define LWE397_SU_INST_45_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_45_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_45_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_45_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_45_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_45_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_45_P3_POINT                  31:30
#define LWE397_SU_INST_45_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_45_P3_POINT_S                      (1)
#define LWE397_SU_INST_45_P3_POINT_T                      (2)

#define LWE397_SU_INST_45_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_45_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_45_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_45_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_45_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_45_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_45_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_45_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_45_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_45_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_45_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_45_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_45_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_45_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_45_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_45_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_45_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_45_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_45_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_45_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_45_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_45_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_45_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_45_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_45_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_45_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_45_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_45_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_45_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_45_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_45_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_45_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_45_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_45_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_45_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_45_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_46  
#define LWE397_SU_INST_46                 (0x32e)
#define LWE397_SU_INST_46_SRC                       1:0
#define LWE397_SU_INST_46_SRC_VPE                 (0)
#define LWE397_SU_INST_46_SRC_Z                   (1)

#define LWE397_SU_INST_46_VC_ROW                    6:3

#define LWE397_SU_INST_46_TRAM_ROW                  14:9

#define LWE397_SU_INST_46_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_46_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_46_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_46_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_46_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_46_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_46_P0_POINT                  19:18
#define LWE397_SU_INST_46_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_46_P0_POINT_S                      (1)
#define LWE397_SU_INST_46_P0_POINT_T                      (2)

#define LWE397_SU_INST_46_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_46_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_46_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_46_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_46_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_46_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_46_P1_POINT                  23:22
#define LWE397_SU_INST_46_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_46_P1_POINT_S                      (1)
#define LWE397_SU_INST_46_P1_POINT_T                      (2)

#define LWE397_SU_INST_46_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_46_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_46_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_46_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_46_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_46_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_46_P2_POINT                  27:26
#define LWE397_SU_INST_46_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_46_P2_POINT_S                      (1)
#define LWE397_SU_INST_46_P2_POINT_T                      (2)

#define LWE397_SU_INST_46_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_46_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_46_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_46_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_46_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_46_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_46_P3_POINT                  31:30
#define LWE397_SU_INST_46_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_46_P3_POINT_S                      (1)
#define LWE397_SU_INST_46_P3_POINT_T                      (2)

#define LWE397_SU_INST_46_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_46_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_46_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_46_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_46_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_46_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_46_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_46_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_46_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_46_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_46_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_46_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_46_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_46_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_46_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_46_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_46_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_46_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_46_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_46_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_46_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_46_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_46_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_46_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_46_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_46_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_46_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_46_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_46_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_46_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_46_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_46_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_46_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_46_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_46_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_46_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_47  
#define LWE397_SU_INST_47                 (0x32f)
#define LWE397_SU_INST_47_SRC                       1:0
#define LWE397_SU_INST_47_SRC_VPE                 (0)
#define LWE397_SU_INST_47_SRC_Z                   (1)

#define LWE397_SU_INST_47_VC_ROW                    6:3

#define LWE397_SU_INST_47_TRAM_ROW                  14:9

#define LWE397_SU_INST_47_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_47_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_47_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_47_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_47_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_47_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_47_P0_POINT                  19:18
#define LWE397_SU_INST_47_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_47_P0_POINT_S                      (1)
#define LWE397_SU_INST_47_P0_POINT_T                      (2)

#define LWE397_SU_INST_47_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_47_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_47_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_47_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_47_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_47_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_47_P1_POINT                  23:22
#define LWE397_SU_INST_47_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_47_P1_POINT_S                      (1)
#define LWE397_SU_INST_47_P1_POINT_T                      (2)

#define LWE397_SU_INST_47_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_47_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_47_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_47_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_47_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_47_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_47_P2_POINT                  27:26
#define LWE397_SU_INST_47_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_47_P2_POINT_S                      (1)
#define LWE397_SU_INST_47_P2_POINT_T                      (2)

#define LWE397_SU_INST_47_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_47_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_47_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_47_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_47_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_47_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_47_P3_POINT                  31:30
#define LWE397_SU_INST_47_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_47_P3_POINT_S                      (1)
#define LWE397_SU_INST_47_P3_POINT_T                      (2)

#define LWE397_SU_INST_47_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_47_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_47_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_47_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_47_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_47_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_47_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_47_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_47_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_47_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_47_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_47_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_47_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_47_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_47_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_47_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_47_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_47_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_47_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_47_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_47_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_47_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_47_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_47_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_47_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_47_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_47_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_47_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_47_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_47_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_47_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_47_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_47_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_47_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_47_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_47_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_48  
#define LWE397_SU_INST_48                 (0x330)
#define LWE397_SU_INST_48_SRC                       1:0
#define LWE397_SU_INST_48_SRC_VPE                 (0)
#define LWE397_SU_INST_48_SRC_Z                   (1)

#define LWE397_SU_INST_48_VC_ROW                    6:3

#define LWE397_SU_INST_48_TRAM_ROW                  14:9

#define LWE397_SU_INST_48_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_48_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_48_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_48_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_48_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_48_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_48_P0_POINT                  19:18
#define LWE397_SU_INST_48_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_48_P0_POINT_S                      (1)
#define LWE397_SU_INST_48_P0_POINT_T                      (2)

#define LWE397_SU_INST_48_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_48_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_48_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_48_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_48_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_48_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_48_P1_POINT                  23:22
#define LWE397_SU_INST_48_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_48_P1_POINT_S                      (1)
#define LWE397_SU_INST_48_P1_POINT_T                      (2)

#define LWE397_SU_INST_48_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_48_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_48_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_48_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_48_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_48_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_48_P2_POINT                  27:26
#define LWE397_SU_INST_48_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_48_P2_POINT_S                      (1)
#define LWE397_SU_INST_48_P2_POINT_T                      (2)

#define LWE397_SU_INST_48_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_48_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_48_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_48_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_48_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_48_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_48_P3_POINT                  31:30
#define LWE397_SU_INST_48_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_48_P3_POINT_S                      (1)
#define LWE397_SU_INST_48_P3_POINT_T                      (2)

#define LWE397_SU_INST_48_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_48_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_48_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_48_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_48_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_48_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_48_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_48_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_48_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_48_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_48_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_48_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_48_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_48_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_48_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_48_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_48_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_48_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_48_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_48_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_48_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_48_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_48_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_48_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_48_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_48_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_48_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_48_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_48_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_48_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_48_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_48_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_48_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_48_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_48_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_48_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_49  
#define LWE397_SU_INST_49                 (0x331)
#define LWE397_SU_INST_49_SRC                       1:0
#define LWE397_SU_INST_49_SRC_VPE                 (0)
#define LWE397_SU_INST_49_SRC_Z                   (1)

#define LWE397_SU_INST_49_VC_ROW                    6:3

#define LWE397_SU_INST_49_TRAM_ROW                  14:9

#define LWE397_SU_INST_49_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_49_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_49_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_49_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_49_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_49_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_49_P0_POINT                  19:18
#define LWE397_SU_INST_49_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_49_P0_POINT_S                      (1)
#define LWE397_SU_INST_49_P0_POINT_T                      (2)

#define LWE397_SU_INST_49_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_49_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_49_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_49_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_49_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_49_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_49_P1_POINT                  23:22
#define LWE397_SU_INST_49_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_49_P1_POINT_S                      (1)
#define LWE397_SU_INST_49_P1_POINT_T                      (2)

#define LWE397_SU_INST_49_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_49_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_49_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_49_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_49_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_49_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_49_P2_POINT                  27:26
#define LWE397_SU_INST_49_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_49_P2_POINT_S                      (1)
#define LWE397_SU_INST_49_P2_POINT_T                      (2)

#define LWE397_SU_INST_49_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_49_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_49_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_49_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_49_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_49_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_49_P3_POINT                  31:30
#define LWE397_SU_INST_49_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_49_P3_POINT_S                      (1)
#define LWE397_SU_INST_49_P3_POINT_T                      (2)

#define LWE397_SU_INST_49_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_49_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_49_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_49_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_49_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_49_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_49_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_49_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_49_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_49_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_49_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_49_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_49_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_49_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_49_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_49_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_49_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_49_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_49_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_49_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_49_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_49_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_49_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_49_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_49_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_49_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_49_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_49_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_49_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_49_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_49_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_49_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_49_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_49_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_49_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_49_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_50  
#define LWE397_SU_INST_50                 (0x332)
#define LWE397_SU_INST_50_SRC                       1:0
#define LWE397_SU_INST_50_SRC_VPE                 (0)
#define LWE397_SU_INST_50_SRC_Z                   (1)

#define LWE397_SU_INST_50_VC_ROW                    6:3

#define LWE397_SU_INST_50_TRAM_ROW                  14:9

#define LWE397_SU_INST_50_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_50_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_50_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_50_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_50_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_50_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_50_P0_POINT                  19:18
#define LWE397_SU_INST_50_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_50_P0_POINT_S                      (1)
#define LWE397_SU_INST_50_P0_POINT_T                      (2)

#define LWE397_SU_INST_50_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_50_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_50_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_50_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_50_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_50_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_50_P1_POINT                  23:22
#define LWE397_SU_INST_50_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_50_P1_POINT_S                      (1)
#define LWE397_SU_INST_50_P1_POINT_T                      (2)

#define LWE397_SU_INST_50_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_50_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_50_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_50_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_50_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_50_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_50_P2_POINT                  27:26
#define LWE397_SU_INST_50_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_50_P2_POINT_S                      (1)
#define LWE397_SU_INST_50_P2_POINT_T                      (2)

#define LWE397_SU_INST_50_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_50_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_50_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_50_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_50_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_50_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_50_P3_POINT                  31:30
#define LWE397_SU_INST_50_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_50_P3_POINT_S                      (1)
#define LWE397_SU_INST_50_P3_POINT_T                      (2)

#define LWE397_SU_INST_50_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_50_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_50_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_50_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_50_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_50_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_50_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_50_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_50_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_50_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_50_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_50_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_50_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_50_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_50_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_50_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_50_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_50_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_50_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_50_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_50_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_50_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_50_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_50_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_50_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_50_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_50_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_50_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_50_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_50_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_50_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_50_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_50_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_50_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_50_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_50_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_51  
#define LWE397_SU_INST_51                 (0x333)
#define LWE397_SU_INST_51_SRC                       1:0
#define LWE397_SU_INST_51_SRC_VPE                 (0)
#define LWE397_SU_INST_51_SRC_Z                   (1)

#define LWE397_SU_INST_51_VC_ROW                    6:3

#define LWE397_SU_INST_51_TRAM_ROW                  14:9

#define LWE397_SU_INST_51_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_51_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_51_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_51_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_51_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_51_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_51_P0_POINT                  19:18
#define LWE397_SU_INST_51_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_51_P0_POINT_S                      (1)
#define LWE397_SU_INST_51_P0_POINT_T                      (2)

#define LWE397_SU_INST_51_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_51_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_51_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_51_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_51_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_51_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_51_P1_POINT                  23:22
#define LWE397_SU_INST_51_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_51_P1_POINT_S                      (1)
#define LWE397_SU_INST_51_P1_POINT_T                      (2)

#define LWE397_SU_INST_51_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_51_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_51_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_51_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_51_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_51_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_51_P2_POINT                  27:26
#define LWE397_SU_INST_51_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_51_P2_POINT_S                      (1)
#define LWE397_SU_INST_51_P2_POINT_T                      (2)

#define LWE397_SU_INST_51_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_51_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_51_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_51_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_51_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_51_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_51_P3_POINT                  31:30
#define LWE397_SU_INST_51_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_51_P3_POINT_S                      (1)
#define LWE397_SU_INST_51_P3_POINT_T                      (2)

#define LWE397_SU_INST_51_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_51_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_51_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_51_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_51_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_51_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_51_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_51_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_51_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_51_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_51_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_51_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_51_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_51_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_51_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_51_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_51_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_51_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_51_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_51_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_51_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_51_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_51_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_51_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_51_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_51_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_51_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_51_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_51_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_51_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_51_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_51_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_51_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_51_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_51_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_51_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_52  
#define LWE397_SU_INST_52                 (0x334)
#define LWE397_SU_INST_52_SRC                       1:0
#define LWE397_SU_INST_52_SRC_VPE                 (0)
#define LWE397_SU_INST_52_SRC_Z                   (1)

#define LWE397_SU_INST_52_VC_ROW                    6:3

#define LWE397_SU_INST_52_TRAM_ROW                  14:9

#define LWE397_SU_INST_52_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_52_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_52_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_52_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_52_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_52_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_52_P0_POINT                  19:18
#define LWE397_SU_INST_52_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_52_P0_POINT_S                      (1)
#define LWE397_SU_INST_52_P0_POINT_T                      (2)

#define LWE397_SU_INST_52_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_52_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_52_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_52_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_52_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_52_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_52_P1_POINT                  23:22
#define LWE397_SU_INST_52_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_52_P1_POINT_S                      (1)
#define LWE397_SU_INST_52_P1_POINT_T                      (2)

#define LWE397_SU_INST_52_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_52_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_52_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_52_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_52_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_52_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_52_P2_POINT                  27:26
#define LWE397_SU_INST_52_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_52_P2_POINT_S                      (1)
#define LWE397_SU_INST_52_P2_POINT_T                      (2)

#define LWE397_SU_INST_52_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_52_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_52_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_52_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_52_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_52_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_52_P3_POINT                  31:30
#define LWE397_SU_INST_52_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_52_P3_POINT_S                      (1)
#define LWE397_SU_INST_52_P3_POINT_T                      (2)

#define LWE397_SU_INST_52_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_52_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_52_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_52_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_52_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_52_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_52_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_52_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_52_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_52_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_52_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_52_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_52_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_52_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_52_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_52_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_52_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_52_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_52_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_52_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_52_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_52_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_52_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_52_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_52_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_52_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_52_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_52_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_52_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_52_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_52_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_52_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_52_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_52_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_52_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_52_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_53  
#define LWE397_SU_INST_53                 (0x335)
#define LWE397_SU_INST_53_SRC                       1:0
#define LWE397_SU_INST_53_SRC_VPE                 (0)
#define LWE397_SU_INST_53_SRC_Z                   (1)

#define LWE397_SU_INST_53_VC_ROW                    6:3

#define LWE397_SU_INST_53_TRAM_ROW                  14:9

#define LWE397_SU_INST_53_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_53_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_53_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_53_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_53_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_53_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_53_P0_POINT                  19:18
#define LWE397_SU_INST_53_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_53_P0_POINT_S                      (1)
#define LWE397_SU_INST_53_P0_POINT_T                      (2)

#define LWE397_SU_INST_53_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_53_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_53_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_53_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_53_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_53_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_53_P1_POINT                  23:22
#define LWE397_SU_INST_53_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_53_P1_POINT_S                      (1)
#define LWE397_SU_INST_53_P1_POINT_T                      (2)

#define LWE397_SU_INST_53_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_53_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_53_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_53_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_53_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_53_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_53_P2_POINT                  27:26
#define LWE397_SU_INST_53_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_53_P2_POINT_S                      (1)
#define LWE397_SU_INST_53_P2_POINT_T                      (2)

#define LWE397_SU_INST_53_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_53_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_53_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_53_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_53_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_53_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_53_P3_POINT                  31:30
#define LWE397_SU_INST_53_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_53_P3_POINT_S                      (1)
#define LWE397_SU_INST_53_P3_POINT_T                      (2)

#define LWE397_SU_INST_53_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_53_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_53_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_53_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_53_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_53_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_53_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_53_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_53_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_53_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_53_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_53_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_53_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_53_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_53_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_53_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_53_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_53_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_53_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_53_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_53_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_53_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_53_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_53_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_53_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_53_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_53_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_53_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_53_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_53_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_53_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_53_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_53_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_53_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_53_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_53_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_54  
#define LWE397_SU_INST_54                 (0x336)
#define LWE397_SU_INST_54_SRC                       1:0
#define LWE397_SU_INST_54_SRC_VPE                 (0)
#define LWE397_SU_INST_54_SRC_Z                   (1)

#define LWE397_SU_INST_54_VC_ROW                    6:3

#define LWE397_SU_INST_54_TRAM_ROW                  14:9

#define LWE397_SU_INST_54_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_54_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_54_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_54_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_54_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_54_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_54_P0_POINT                  19:18
#define LWE397_SU_INST_54_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_54_P0_POINT_S                      (1)
#define LWE397_SU_INST_54_P0_POINT_T                      (2)

#define LWE397_SU_INST_54_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_54_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_54_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_54_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_54_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_54_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_54_P1_POINT                  23:22
#define LWE397_SU_INST_54_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_54_P1_POINT_S                      (1)
#define LWE397_SU_INST_54_P1_POINT_T                      (2)

#define LWE397_SU_INST_54_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_54_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_54_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_54_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_54_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_54_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_54_P2_POINT                  27:26
#define LWE397_SU_INST_54_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_54_P2_POINT_S                      (1)
#define LWE397_SU_INST_54_P2_POINT_T                      (2)

#define LWE397_SU_INST_54_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_54_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_54_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_54_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_54_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_54_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_54_P3_POINT                  31:30
#define LWE397_SU_INST_54_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_54_P3_POINT_S                      (1)
#define LWE397_SU_INST_54_P3_POINT_T                      (2)

#define LWE397_SU_INST_54_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_54_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_54_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_54_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_54_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_54_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_54_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_54_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_54_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_54_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_54_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_54_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_54_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_54_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_54_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_54_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_54_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_54_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_54_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_54_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_54_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_54_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_54_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_54_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_54_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_54_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_54_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_54_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_54_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_54_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_54_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_54_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_54_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_54_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_54_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_54_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_55  
#define LWE397_SU_INST_55                 (0x337)
#define LWE397_SU_INST_55_SRC                       1:0
#define LWE397_SU_INST_55_SRC_VPE                 (0)
#define LWE397_SU_INST_55_SRC_Z                   (1)

#define LWE397_SU_INST_55_VC_ROW                    6:3

#define LWE397_SU_INST_55_TRAM_ROW                  14:9

#define LWE397_SU_INST_55_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_55_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_55_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_55_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_55_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_55_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_55_P0_POINT                  19:18
#define LWE397_SU_INST_55_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_55_P0_POINT_S                      (1)
#define LWE397_SU_INST_55_P0_POINT_T                      (2)

#define LWE397_SU_INST_55_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_55_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_55_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_55_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_55_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_55_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_55_P1_POINT                  23:22
#define LWE397_SU_INST_55_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_55_P1_POINT_S                      (1)
#define LWE397_SU_INST_55_P1_POINT_T                      (2)

#define LWE397_SU_INST_55_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_55_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_55_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_55_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_55_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_55_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_55_P2_POINT                  27:26
#define LWE397_SU_INST_55_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_55_P2_POINT_S                      (1)
#define LWE397_SU_INST_55_P2_POINT_T                      (2)

#define LWE397_SU_INST_55_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_55_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_55_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_55_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_55_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_55_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_55_P3_POINT                  31:30
#define LWE397_SU_INST_55_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_55_P3_POINT_S                      (1)
#define LWE397_SU_INST_55_P3_POINT_T                      (2)

#define LWE397_SU_INST_55_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_55_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_55_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_55_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_55_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_55_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_55_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_55_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_55_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_55_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_55_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_55_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_55_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_55_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_55_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_55_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_55_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_55_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_55_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_55_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_55_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_55_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_55_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_55_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_55_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_55_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_55_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_55_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_55_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_55_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_55_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_55_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_55_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_55_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_55_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_55_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_56  
#define LWE397_SU_INST_56                 (0x338)
#define LWE397_SU_INST_56_SRC                       1:0
#define LWE397_SU_INST_56_SRC_VPE                 (0)
#define LWE397_SU_INST_56_SRC_Z                   (1)

#define LWE397_SU_INST_56_VC_ROW                    6:3

#define LWE397_SU_INST_56_TRAM_ROW                  14:9

#define LWE397_SU_INST_56_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_56_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_56_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_56_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_56_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_56_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_56_P0_POINT                  19:18
#define LWE397_SU_INST_56_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_56_P0_POINT_S                      (1)
#define LWE397_SU_INST_56_P0_POINT_T                      (2)

#define LWE397_SU_INST_56_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_56_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_56_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_56_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_56_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_56_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_56_P1_POINT                  23:22
#define LWE397_SU_INST_56_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_56_P1_POINT_S                      (1)
#define LWE397_SU_INST_56_P1_POINT_T                      (2)

#define LWE397_SU_INST_56_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_56_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_56_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_56_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_56_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_56_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_56_P2_POINT                  27:26
#define LWE397_SU_INST_56_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_56_P2_POINT_S                      (1)
#define LWE397_SU_INST_56_P2_POINT_T                      (2)

#define LWE397_SU_INST_56_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_56_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_56_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_56_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_56_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_56_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_56_P3_POINT                  31:30
#define LWE397_SU_INST_56_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_56_P3_POINT_S                      (1)
#define LWE397_SU_INST_56_P3_POINT_T                      (2)

#define LWE397_SU_INST_56_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_56_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_56_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_56_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_56_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_56_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_56_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_56_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_56_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_56_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_56_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_56_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_56_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_56_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_56_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_56_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_56_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_56_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_56_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_56_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_56_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_56_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_56_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_56_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_56_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_56_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_56_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_56_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_56_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_56_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_56_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_56_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_56_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_56_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_56_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_56_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_57  
#define LWE397_SU_INST_57                 (0x339)
#define LWE397_SU_INST_57_SRC                       1:0
#define LWE397_SU_INST_57_SRC_VPE                 (0)
#define LWE397_SU_INST_57_SRC_Z                   (1)

#define LWE397_SU_INST_57_VC_ROW                    6:3

#define LWE397_SU_INST_57_TRAM_ROW                  14:9

#define LWE397_SU_INST_57_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_57_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_57_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_57_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_57_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_57_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_57_P0_POINT                  19:18
#define LWE397_SU_INST_57_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_57_P0_POINT_S                      (1)
#define LWE397_SU_INST_57_P0_POINT_T                      (2)

#define LWE397_SU_INST_57_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_57_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_57_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_57_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_57_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_57_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_57_P1_POINT                  23:22
#define LWE397_SU_INST_57_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_57_P1_POINT_S                      (1)
#define LWE397_SU_INST_57_P1_POINT_T                      (2)

#define LWE397_SU_INST_57_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_57_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_57_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_57_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_57_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_57_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_57_P2_POINT                  27:26
#define LWE397_SU_INST_57_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_57_P2_POINT_S                      (1)
#define LWE397_SU_INST_57_P2_POINT_T                      (2)

#define LWE397_SU_INST_57_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_57_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_57_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_57_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_57_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_57_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_57_P3_POINT                  31:30
#define LWE397_SU_INST_57_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_57_P3_POINT_S                      (1)
#define LWE397_SU_INST_57_P3_POINT_T                      (2)

#define LWE397_SU_INST_57_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_57_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_57_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_57_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_57_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_57_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_57_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_57_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_57_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_57_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_57_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_57_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_57_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_57_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_57_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_57_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_57_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_57_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_57_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_57_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_57_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_57_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_57_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_57_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_57_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_57_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_57_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_57_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_57_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_57_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_57_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_57_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_57_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_57_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_57_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_57_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_58  
#define LWE397_SU_INST_58                 (0x33a)
#define LWE397_SU_INST_58_SRC                       1:0
#define LWE397_SU_INST_58_SRC_VPE                 (0)
#define LWE397_SU_INST_58_SRC_Z                   (1)

#define LWE397_SU_INST_58_VC_ROW                    6:3

#define LWE397_SU_INST_58_TRAM_ROW                  14:9

#define LWE397_SU_INST_58_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_58_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_58_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_58_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_58_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_58_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_58_P0_POINT                  19:18
#define LWE397_SU_INST_58_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_58_P0_POINT_S                      (1)
#define LWE397_SU_INST_58_P0_POINT_T                      (2)

#define LWE397_SU_INST_58_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_58_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_58_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_58_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_58_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_58_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_58_P1_POINT                  23:22
#define LWE397_SU_INST_58_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_58_P1_POINT_S                      (1)
#define LWE397_SU_INST_58_P1_POINT_T                      (2)

#define LWE397_SU_INST_58_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_58_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_58_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_58_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_58_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_58_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_58_P2_POINT                  27:26
#define LWE397_SU_INST_58_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_58_P2_POINT_S                      (1)
#define LWE397_SU_INST_58_P2_POINT_T                      (2)

#define LWE397_SU_INST_58_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_58_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_58_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_58_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_58_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_58_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_58_P3_POINT                  31:30
#define LWE397_SU_INST_58_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_58_P3_POINT_S                      (1)
#define LWE397_SU_INST_58_P3_POINT_T                      (2)

#define LWE397_SU_INST_58_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_58_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_58_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_58_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_58_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_58_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_58_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_58_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_58_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_58_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_58_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_58_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_58_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_58_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_58_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_58_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_58_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_58_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_58_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_58_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_58_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_58_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_58_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_58_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_58_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_58_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_58_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_58_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_58_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_58_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_58_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_58_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_58_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_58_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_58_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_58_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_59  
#define LWE397_SU_INST_59                 (0x33b)
#define LWE397_SU_INST_59_SRC                       1:0
#define LWE397_SU_INST_59_SRC_VPE                 (0)
#define LWE397_SU_INST_59_SRC_Z                   (1)

#define LWE397_SU_INST_59_VC_ROW                    6:3

#define LWE397_SU_INST_59_TRAM_ROW                  14:9

#define LWE397_SU_INST_59_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_59_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_59_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_59_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_59_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_59_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_59_P0_POINT                  19:18
#define LWE397_SU_INST_59_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_59_P0_POINT_S                      (1)
#define LWE397_SU_INST_59_P0_POINT_T                      (2)

#define LWE397_SU_INST_59_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_59_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_59_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_59_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_59_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_59_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_59_P1_POINT                  23:22
#define LWE397_SU_INST_59_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_59_P1_POINT_S                      (1)
#define LWE397_SU_INST_59_P1_POINT_T                      (2)

#define LWE397_SU_INST_59_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_59_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_59_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_59_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_59_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_59_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_59_P2_POINT                  27:26
#define LWE397_SU_INST_59_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_59_P2_POINT_S                      (1)
#define LWE397_SU_INST_59_P2_POINT_T                      (2)

#define LWE397_SU_INST_59_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_59_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_59_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_59_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_59_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_59_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_59_P3_POINT                  31:30
#define LWE397_SU_INST_59_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_59_P3_POINT_S                      (1)
#define LWE397_SU_INST_59_P3_POINT_T                      (2)

#define LWE397_SU_INST_59_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_59_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_59_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_59_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_59_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_59_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_59_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_59_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_59_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_59_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_59_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_59_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_59_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_59_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_59_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_59_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_59_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_59_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_59_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_59_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_59_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_59_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_59_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_59_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_59_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_59_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_59_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_59_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_59_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_59_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_59_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_59_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_59_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_59_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_59_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_59_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_60  
#define LWE397_SU_INST_60                 (0x33c)
#define LWE397_SU_INST_60_SRC                       1:0
#define LWE397_SU_INST_60_SRC_VPE                 (0)
#define LWE397_SU_INST_60_SRC_Z                   (1)

#define LWE397_SU_INST_60_VC_ROW                    6:3

#define LWE397_SU_INST_60_TRAM_ROW                  14:9

#define LWE397_SU_INST_60_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_60_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_60_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_60_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_60_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_60_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_60_P0_POINT                  19:18
#define LWE397_SU_INST_60_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_60_P0_POINT_S                      (1)
#define LWE397_SU_INST_60_P0_POINT_T                      (2)

#define LWE397_SU_INST_60_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_60_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_60_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_60_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_60_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_60_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_60_P1_POINT                  23:22
#define LWE397_SU_INST_60_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_60_P1_POINT_S                      (1)
#define LWE397_SU_INST_60_P1_POINT_T                      (2)

#define LWE397_SU_INST_60_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_60_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_60_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_60_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_60_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_60_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_60_P2_POINT                  27:26
#define LWE397_SU_INST_60_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_60_P2_POINT_S                      (1)
#define LWE397_SU_INST_60_P2_POINT_T                      (2)

#define LWE397_SU_INST_60_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_60_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_60_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_60_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_60_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_60_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_60_P3_POINT                  31:30
#define LWE397_SU_INST_60_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_60_P3_POINT_S                      (1)
#define LWE397_SU_INST_60_P3_POINT_T                      (2)

#define LWE397_SU_INST_60_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_60_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_60_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_60_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_60_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_60_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_60_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_60_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_60_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_60_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_60_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_60_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_60_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_60_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_60_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_60_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_60_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_60_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_60_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_60_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_60_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_60_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_60_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_60_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_60_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_60_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_60_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_60_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_60_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_60_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_60_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_60_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_60_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_60_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_60_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_60_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_61  
#define LWE397_SU_INST_61                 (0x33d)
#define LWE397_SU_INST_61_SRC                       1:0
#define LWE397_SU_INST_61_SRC_VPE                 (0)
#define LWE397_SU_INST_61_SRC_Z                   (1)

#define LWE397_SU_INST_61_VC_ROW                    6:3

#define LWE397_SU_INST_61_TRAM_ROW                  14:9

#define LWE397_SU_INST_61_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_61_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_61_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_61_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_61_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_61_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_61_P0_POINT                  19:18
#define LWE397_SU_INST_61_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_61_P0_POINT_S                      (1)
#define LWE397_SU_INST_61_P0_POINT_T                      (2)

#define LWE397_SU_INST_61_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_61_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_61_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_61_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_61_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_61_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_61_P1_POINT                  23:22
#define LWE397_SU_INST_61_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_61_P1_POINT_S                      (1)
#define LWE397_SU_INST_61_P1_POINT_T                      (2)

#define LWE397_SU_INST_61_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_61_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_61_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_61_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_61_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_61_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_61_P2_POINT                  27:26
#define LWE397_SU_INST_61_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_61_P2_POINT_S                      (1)
#define LWE397_SU_INST_61_P2_POINT_T                      (2)

#define LWE397_SU_INST_61_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_61_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_61_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_61_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_61_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_61_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_61_P3_POINT                  31:30
#define LWE397_SU_INST_61_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_61_P3_POINT_S                      (1)
#define LWE397_SU_INST_61_P3_POINT_T                      (2)

#define LWE397_SU_INST_61_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_61_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_61_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_61_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_61_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_61_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_61_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_61_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_61_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_61_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_61_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_61_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_61_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_61_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_61_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_61_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_61_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_61_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_61_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_61_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_61_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_61_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_61_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_61_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_61_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_61_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_61_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_61_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_61_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_61_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_61_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_61_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_61_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_61_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_61_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_61_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_62  
#define LWE397_SU_INST_62                 (0x33e)
#define LWE397_SU_INST_62_SRC                       1:0
#define LWE397_SU_INST_62_SRC_VPE                 (0)
#define LWE397_SU_INST_62_SRC_Z                   (1)

#define LWE397_SU_INST_62_VC_ROW                    6:3

#define LWE397_SU_INST_62_TRAM_ROW                  14:9

#define LWE397_SU_INST_62_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_62_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_62_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_62_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_62_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_62_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_62_P0_POINT                  19:18
#define LWE397_SU_INST_62_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_62_P0_POINT_S                      (1)
#define LWE397_SU_INST_62_P0_POINT_T                      (2)

#define LWE397_SU_INST_62_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_62_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_62_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_62_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_62_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_62_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_62_P1_POINT                  23:22
#define LWE397_SU_INST_62_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_62_P1_POINT_S                      (1)
#define LWE397_SU_INST_62_P1_POINT_T                      (2)

#define LWE397_SU_INST_62_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_62_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_62_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_62_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_62_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_62_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_62_P2_POINT                  27:26
#define LWE397_SU_INST_62_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_62_P2_POINT_S                      (1)
#define LWE397_SU_INST_62_P2_POINT_T                      (2)

#define LWE397_SU_INST_62_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_62_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_62_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_62_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_62_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_62_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_62_P3_POINT                  31:30
#define LWE397_SU_INST_62_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_62_P3_POINT_S                      (1)
#define LWE397_SU_INST_62_P3_POINT_T                      (2)

#define LWE397_SU_INST_62_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_62_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_62_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_62_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_62_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_62_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_62_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_62_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_62_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_62_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_62_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_62_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_62_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_62_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_62_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_62_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_62_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_62_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_62_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_62_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_62_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_62_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_62_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_62_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_62_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_62_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_62_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_62_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_62_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_62_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_62_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_62_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_62_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_62_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_62_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_62_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_INST_63  
#define LWE397_SU_INST_63                 (0x33f)
#define LWE397_SU_INST_63_SRC                       1:0
#define LWE397_SU_INST_63_SRC_VPE                 (0)
#define LWE397_SU_INST_63_SRC_Z                   (1)

#define LWE397_SU_INST_63_VC_ROW                    6:3

#define LWE397_SU_INST_63_TRAM_ROW                  14:9

#define LWE397_SU_INST_63_P0_LINE_WIDTH                     16:16
#define LWE397_SU_INST_63_P0_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_63_P0_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_63_P0_LINE_LENGTH                    17:17
#define LWE397_SU_INST_63_P0_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_63_P0_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_63_P0_POINT                  19:18
#define LWE397_SU_INST_63_P0_POINT_DISABLE                        (0)
#define LWE397_SU_INST_63_P0_POINT_S                      (1)
#define LWE397_SU_INST_63_P0_POINT_T                      (2)

#define LWE397_SU_INST_63_P1_LINE_WIDTH                     20:20
#define LWE397_SU_INST_63_P1_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_63_P1_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_63_P1_LINE_LENGTH                    21:21
#define LWE397_SU_INST_63_P1_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_63_P1_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_63_P1_POINT                  23:22
#define LWE397_SU_INST_63_P1_POINT_DISABLE                        (0)
#define LWE397_SU_INST_63_P1_POINT_S                      (1)
#define LWE397_SU_INST_63_P1_POINT_T                      (2)

#define LWE397_SU_INST_63_P2_LINE_WIDTH                     24:24
#define LWE397_SU_INST_63_P2_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_63_P2_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_63_P2_LINE_LENGTH                    25:25
#define LWE397_SU_INST_63_P2_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_63_P2_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_63_P2_POINT                  27:26
#define LWE397_SU_INST_63_P2_POINT_DISABLE                        (0)
#define LWE397_SU_INST_63_P2_POINT_S                      (1)
#define LWE397_SU_INST_63_P2_POINT_T                      (2)

#define LWE397_SU_INST_63_P3_LINE_WIDTH                     28:28
#define LWE397_SU_INST_63_P3_LINE_WIDTH_CONST                     (0)
#define LWE397_SU_INST_63_P3_LINE_WIDTH_VARYING                   (1)

#define LWE397_SU_INST_63_P3_LINE_LENGTH                    29:29
#define LWE397_SU_INST_63_P3_LINE_LENGTH_VARYING                  (0)
#define LWE397_SU_INST_63_P3_LINE_LENGTH_CONST                    (1)

#define LWE397_SU_INST_63_P3_POINT                  31:30
#define LWE397_SU_INST_63_P3_POINT_DISABLE                        (0)
#define LWE397_SU_INST_63_P3_POINT_S                      (1)
#define LWE397_SU_INST_63_P3_POINT_T                      (2)

#define LWE397_SU_INST_63_P0_TRAM_COL                       1:0

#define LWE397_SU_INST_63_P0_TRAM_FMT                       3:2
#define LWE397_SU_INST_63_P0_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_63_P0_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_63_P0_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_63_P0_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_63_P1_TRAM_COL                       5:4

#define LWE397_SU_INST_63_P1_TRAM_FMT                       7:6
#define LWE397_SU_INST_63_P1_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_63_P1_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_63_P1_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_63_P1_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_63_P2_TRAM_COL                       9:8

#define LWE397_SU_INST_63_P2_TRAM_FMT                       11:10
#define LWE397_SU_INST_63_P2_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_63_P2_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_63_P2_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_63_P2_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_63_P3_TRAM_COL                       13:12

#define LWE397_SU_INST_63_P3_TRAM_FMT                       15:14
#define LWE397_SU_INST_63_P3_TRAM_FMT_NOP                 (0)
#define LWE397_SU_INST_63_P3_TRAM_FMT_LP_LO                       (1)
#define LWE397_SU_INST_63_P3_TRAM_FMT_LP_HI                       (2)
#define LWE397_SU_INST_63_P3_TRAM_FMT_HP                  (3)

#define LWE397_SU_INST_63_P0_TRI_SHADE_MODE                 16:16
#define LWE397_SU_INST_63_P0_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_63_P0_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_63_P1_TRI_SHADE_MODE                 17:17
#define LWE397_SU_INST_63_P1_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_63_P1_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_63_P2_TRI_SHADE_MODE                 18:18
#define LWE397_SU_INST_63_P2_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_63_P2_TRI_SHADE_MODE_FLAT                  (1)

#define LWE397_SU_INST_63_P3_TRI_SHADE_MODE                 19:19
#define LWE397_SU_INST_63_P3_TRI_SHADE_MODE_SMOOTH                        (0)
#define LWE397_SU_INST_63_P3_TRI_SHADE_MODE_FLAT                  (1)


// Register LWE397_SU_DRAW_POINT_0  
#define LWE397_SU_DRAW_POINT_0                    (0x340)
#define LWE397_SU_DRAW_POINT_0_V0_LOAD                      24:24

#define LWE397_SU_DRAW_POINT_0_V0                   3:0


// Register LWE397_SU_DRAW_LINE_0  
#define LWE397_SU_DRAW_LINE_0                     (0x341)
#define LWE397_SU_DRAW_LINE_0_V1_LOAD                       25:25

#define LWE397_SU_DRAW_LINE_0_V0_LOAD                       24:24

#define LWE397_SU_DRAW_LINE_0_V1                    11:8

#define LWE397_SU_DRAW_LINE_0_V0                    3:0


// Register LWE397_SU_DRAW_TRI_0  
#define LWE397_SU_DRAW_TRI_0                      (0x342)
#define LWE397_SU_DRAW_TRI_0_V2_LOAD                        26:26

#define LWE397_SU_DRAW_TRI_0_V1_LOAD                        25:25

#define LWE397_SU_DRAW_TRI_0_V0_LOAD                        24:24

#define LWE397_SU_DRAW_TRI_0_V2                     19:16

#define LWE397_SU_DRAW_TRI_0_V1                     11:8

#define LWE397_SU_DRAW_TRI_0_V0                     3:0


// Register LWE397_SU_PARAM_0  
#define LWE397_SU_PARAM_0                 (0x343)
#define LWE397_SU_PARAM_0_FIRST_INST                        4:0

#define LWE397_SU_PARAM_0_LAST_INST                 9:5

#define LWE397_SU_PARAM_0_FRONT_FACE                        15:15
#define LWE397_SU_PARAM_0_FRONT_FACE_POS                  (0)
#define LWE397_SU_PARAM_0_FRONT_FACE_NEG                  (1)

#define LWE397_SU_PARAM_0_LWLL                      17:16
#define LWE397_SU_PARAM_0_LWLL_NONE                       (0)
#define LWE397_SU_PARAM_0_LWLL_POS                        (1)
#define LWE397_SU_PARAM_0_LWLL_NEG                        (2)
#define LWE397_SU_PARAM_0_LWLL_BOTH                       (3)

#define LWE397_SU_PARAM_0_SUBPIX_XOFF                       23:18

#define LWE397_SU_PARAM_0_SUBPIX_YOFF                       29:24

#define LWE397_SU_PARAM_0_TRANSPOSE_XY                      30:30
#define LWE397_SU_PARAM_0_TRANSPOSE_XY_DISABLE                    (0)
#define LWE397_SU_PARAM_0_TRANSPOSE_XY_ENABLE                     (1)

#define LWE397_SU_PARAM_0_CLIP_ENABLE                       31:31
#define LWE397_SU_PARAM_0_CLIP_ENABLE_DISABLE                     (0)
#define LWE397_SU_PARAM_0_CLIP_ENABLE_ENABLE                      (1)


// Register LWE397_SU_ZBIAS_0  
#define LWE397_SU_ZBIAS_0                 (0x344)
#define LWE397_SU_ZBIAS_0_VAL                       21:0


// Register LWE397_SU_ZFACTOR_0  
#define LWE397_SU_ZFACTOR_0                       (0x345)
#define LWE397_SU_ZFACTOR_0_VAL                     31:0


// Register LWE397_SU_POINT_PARAM_0  
#define LWE397_SU_POINT_PARAM_0                   (0x346)
#define LWE397_SU_POINT_PARAM_0_VTX_OFFS                    3:0

#define LWE397_SU_POINT_PARAM_0_ATTR_COMP                   9:8
#define LWE397_SU_POINT_PARAM_0_ATTR_COMP_X                       (0)
#define LWE397_SU_POINT_PARAM_0_ATTR_COMP_Y                       (1)
#define LWE397_SU_POINT_PARAM_0_ATTR_COMP_Z                       (2)
#define LWE397_SU_POINT_PARAM_0_ATTR_COMP_W                       (3)

#define LWE397_SU_POINT_PARAM_0_POINT_SIZE_MODE                     11:10
#define LWE397_SU_POINT_PARAM_0_POINT_SIZE_MODE_FIXED                     (0)
#define LWE397_SU_POINT_PARAM_0_POINT_SIZE_MODE_PSIZE                     (1)
#define LWE397_SU_POINT_PARAM_0_POINT_SIZE_MODE_SHEAR                     (2)

#define LWE397_SU_POINT_PARAM_0_CLIP_MODE                   12:12
#define LWE397_SU_POINT_PARAM_0_CLIP_MODE_SCREEN                  (0)
#define LWE397_SU_POINT_PARAM_0_CLIP_MODE_GUARDBAND                       (1)


// Register LWE397_SU_POINT_WIDTH_2_0  
#define LWE397_SU_POINT_WIDTH_2_0                 (0x347)
#define LWE397_SU_POINT_WIDTH_2_0_VAL                       31:0


// Register LWE397_SU_POINT_MAX_S_0  
#define LWE397_SU_POINT_MAX_S_0                   (0x348)
#define LWE397_SU_POINT_MAX_S_0_VAL                 31:0


// Register LWE397_SU_POINT_MAX_T_0  
#define LWE397_SU_POINT_MAX_T_0                   (0x349)
#define LWE397_SU_POINT_MAX_T_0_VAL                 31:0


// Register LWE397_SU_POINT_MIN_S_0  
#define LWE397_SU_POINT_MIN_S_0                   (0x34a)
#define LWE397_SU_POINT_MIN_S_0_VAL                 31:0


// Register LWE397_SU_POINT_MIN_T_0  
#define LWE397_SU_POINT_MIN_T_0                   (0x34b)
#define LWE397_SU_POINT_MIN_T_0_VAL                 31:0


// Register LWE397_SU_LINE_PARAM_0  
#define LWE397_SU_LINE_PARAM_0                    (0x34c)
#define LWE397_SU_LINE_PARAM_0_LWT                  0:0
#define LWE397_SU_LINE_PARAM_0_LWT_FRENCH                 (0)
#define LWE397_SU_LINE_PARAM_0_LWT_SQUARE                 (1)

#define LWE397_SU_LINE_PARAM_0_CLIP_MODE                    1:1
#define LWE397_SU_LINE_PARAM_0_CLIP_MODE_SCREEN                   (0)
#define LWE397_SU_LINE_PARAM_0_CLIP_MODE_GUARDBAND                        (1)


// Register LWE397_SU_LINE_WIDTH_2_0  
#define LWE397_SU_LINE_WIDTH_2_0                  (0x34d)
#define LWE397_SU_LINE_WIDTH_2_0_VAL                        31:0


// Register LWE397_SU_LINE_MAX_ATTR_W_0  
#define LWE397_SU_LINE_MAX_ATTR_W_0                       (0x34e)
#define LWE397_SU_LINE_MAX_ATTR_W_0_VAL                     31:0


// Register LWE397_SU_LINE_MIN_ATTR_W_0  
#define LWE397_SU_LINE_MIN_ATTR_W_0                       (0x34f)
#define LWE397_SU_LINE_MIN_ATTR_W_0_VAL                     31:0

#define LWE397_SETUP_VTX_XY_WIDTH 12

// Register LWE397_SU_SCISSOR_X_0  
#define LWE397_SU_SCISSOR_X_0                     (0x350)
#define LWE397_SU_SCISSOR_X_0_MAX                   12:0

#define LWE397_SU_SCISSOR_X_0_MIN                   27:16


// Register LWE397_SU_SCISSOR_Y_0  
#define LWE397_SU_SCISSOR_Y_0                     (0x351)
#define LWE397_SU_SCISSOR_Y_0_MAX                   12:0

#define LWE397_SU_SCISSOR_Y_0_MIN                   27:16


// Register LWE397_SU_VIEWPORT_X_0  
#define LWE397_SU_VIEWPORT_X_0                    (0x352)
#define LWE397_SU_VIEWPORT_X_0_VAL                  31:0


// Register LWE397_SU_VIEWPORT_Y_0  
#define LWE397_SU_VIEWPORT_Y_0                    (0x353)
#define LWE397_SU_VIEWPORT_Y_0_VAL                  31:0


// Register LWE397_SU_VIEWPORT_Z_0  
#define LWE397_SU_VIEWPORT_Z_0                    (0x354)
#define LWE397_SU_VIEWPORT_Z_0_VAL                  31:0


// Register LWE397_SU_VIEWPORT_W_0  
#define LWE397_SU_VIEWPORT_W_0                    (0x355)
#define LWE397_SU_VIEWPORT_W_0_VAL                  31:0


// Register LWE397_SU_VIEWPORT_H_0  
#define LWE397_SU_VIEWPORT_H_0                    (0x356)
#define LWE397_SU_VIEWPORT_H_0_VAL                  31:0


// Register LWE397_SU_VIEWPORT_D_0  
#define LWE397_SU_VIEWPORT_D_0                    (0x357)
#define LWE397_SU_VIEWPORT_D_0_VAL                  31:0


// Register LWE397_SU_GUARDBAND_W_0  
#define LWE397_SU_GUARDBAND_W_0                   (0x358)
#define LWE397_SU_GUARDBAND_W_0_VAL                 31:0


// Register LWE397_SU_GUARDBAND_H_0  
#define LWE397_SU_GUARDBAND_H_0                   (0x359)
#define LWE397_SU_GUARDBAND_H_0_VAL                 31:0


// Register LWE397_SU_GUARDBAND_D_0  
#define LWE397_SU_GUARDBAND_D_0                   (0x35a)
#define LWE397_SU_GUARDBAND_D_0_VAL                 31:0


// Register LWE397_SU_UCPLANE_0  
#define LWE397_SU_UCPLANE_0                       (0x35b)
#define LWE397_SU_UCPLANE_0_VTX_OFFS                        3:0

#define LWE397_SU_UCPLANE_0_ATTR_COMP                       9:8
#define LWE397_SU_UCPLANE_0_ATTR_COMP_X                   (0)
#define LWE397_SU_UCPLANE_0_ATTR_COMP_Y                   (1)
#define LWE397_SU_UCPLANE_0_ATTR_COMP_Z                   (2)
#define LWE397_SU_UCPLANE_0_ATTR_COMP_W                   (3)

#define LWE397_SU_UCPLANE_0_ENABLE                  10:10
#define LWE397_SU_UCPLANE_0_ENABLE_DISABLED                       (0)
#define LWE397_SU_UCPLANE_0_ENABLE_ENABLED                        (1)


// Register LWE397_SU_UCPLANE  
#define LWE397_SU_UCPLANE                 (0x35b)
#define LWE397_SU_UCPLANE_VTX_OFFS                  3:0

#define LWE397_SU_UCPLANE_ATTR_COMP                 9:8
#define LWE397_SU_UCPLANE_ATTR_COMP_X                     (0)
#define LWE397_SU_UCPLANE_ATTR_COMP_Y                     (1)
#define LWE397_SU_UCPLANE_ATTR_COMP_Z                     (2)
#define LWE397_SU_UCPLANE_ATTR_COMP_W                     (3)

#define LWE397_SU_UCPLANE_ENABLE                    10:10
#define LWE397_SU_UCPLANE_ENABLE_DISABLED                 (0)
#define LWE397_SU_UCPLANE_ENABLE_ENABLED                  (1)









// Register LWE397_SU_CLKEN_OVERRIDE_0  
#define LWE397_SU_CLKEN_OVERRIDE_0                        (0x363)
#define LWE397_SU_CLKEN_OVERRIDE_0_GR3D_SETUPDPCLK_CLKEN_OVR                        0:0
#define LWE397_SU_CLKEN_OVERRIDE_0_GR3D_SETUPDPCLK_CLKEN_OVR_CLK_GATED                    (0)
#define LWE397_SU_CLKEN_OVERRIDE_0_GR3D_SETUPDPCLK_CLKEN_OVR_CLK_ALWAYS_ON                        (1)


// Register LWE397_SU_CLIP_CLKEN_OVERRIDE_0  
#define LWE397_SU_CLIP_CLKEN_OVERRIDE_0                   (0x364)
#define LWE397_SU_CLIP_CLKEN_OVERRIDE_0_CLIPCGCLK_CLKEN_OVR                 0:0
#define LWE397_SU_CLIP_CLKEN_OVERRIDE_0_CLIPCGCLK_CLKEN_OVR_CLK_GATED                     (0)
#define LWE397_SU_CLIP_CLKEN_OVERRIDE_0_CLIPCGCLK_CLKEN_OVR_CLK_ALWAYS_ON                 (1)

#define LWE397_SU_CLIP_CLKEN_OVERRIDE_0_CLIPCCCLK_CLKEN_OVR                 1:1
#define LWE397_SU_CLIP_CLKEN_OVERRIDE_0_CLIPCCCLK_CLKEN_OVR_CLK_GATED                     (0)
#define LWE397_SU_CLIP_CLKEN_OVERRIDE_0_CLIPCCCLK_CLKEN_OVR_CLK_ALWAYS_ON                 (1)

#define LWE397_SU_CLIP_CLKEN_OVERRIDE_0_CLIPCPCLK_CLKEN_OVR                 2:2
#define LWE397_SU_CLIP_CLKEN_OVERRIDE_0_CLIPCPCLK_CLKEN_OVR_CLK_GATED                     (0)
#define LWE397_SU_CLIP_CLKEN_OVERRIDE_0_CLIPCPCLK_CLKEN_OVR_CLK_ALWAYS_ON                 (1)

#define LWE397_SU_CLIP_CLKEN_OVERRIDE_0_CLIPVPCLK_CLKEN_OVR                 3:3
#define LWE397_SU_CLIP_CLKEN_OVERRIDE_0_CLIPVPCLK_CLKEN_OVR_CLK_GATED                     (0)
#define LWE397_SU_CLIP_CLKEN_OVERRIDE_0_CLIPVPCLK_CLKEN_OVR_CLK_ALWAYS_ON                 (1)

#define LWE397_SU_CLIP_CLKEN_OVERRIDE_0_CLIPARCLK_CLKEN_OVR                 4:4
#define LWE397_SU_CLIP_CLKEN_OVERRIDE_0_CLIPARCLK_CLKEN_OVR_CLK_GATED                     (0)
#define LWE397_SU_CLIP_CLKEN_OVERRIDE_0_CLIPARCLK_CLKEN_OVR_CLK_ALWAYS_ON                 (1)

#define LWE397_SU_CLIP_CLKEN_OVERRIDE_0_CLIPPACLK_CLKEN_OVR                 5:5
#define LWE397_SU_CLIP_CLKEN_OVERRIDE_0_CLIPPACLK_CLKEN_OVR_CLK_GATED                     (0)
#define LWE397_SU_CLIP_CLKEN_OVERRIDE_0_CLIPPACLK_CLKEN_OVR_CLK_ALWAYS_ON                 (1)

#define LWE397_SU_CLIP_CLKEN_OVERRIDE_0_CLIPDZCLK_CLKEN_OVR                 6:6
#define LWE397_SU_CLIP_CLKEN_OVERRIDE_0_CLIPDZCLK_CLKEN_OVR_CLK_GATED                     (0)
#define LWE397_SU_CLIP_CLKEN_OVERRIDE_0_CLIPDZCLK_CLKEN_OVR_CLK_ALWAYS_ON                 (1)

#define LWE397_SU_CLIP_CLKEN_OVERRIDE_0_CLIPTICLK_CLKEN_OVR                 7:7
#define LWE397_SU_CLIP_CLKEN_OVERRIDE_0_CLIPTICLK_CLKEN_OVR_CLK_GATED                     (0)
#define LWE397_SU_CLIP_CLKEN_OVERRIDE_0_CLIPTICLK_CLKEN_OVR_CLK_ALWAYS_ON                 (1)

#define LWE397_SU_CLIP_CLKEN_OVERRIDE_0_CLIPSICLK_CLKEN_OVR                 8:8
#define LWE397_SU_CLIP_CLKEN_OVERRIDE_0_CLIPSICLK_CLKEN_OVR_CLK_GATED                     (0)
#define LWE397_SU_CLIP_CLKEN_OVERRIDE_0_CLIPSICLK_CLKEN_OVR_CLK_ALWAYS_ON                 (1)


// Register LWE397_SU_OUTER_SLI_SCISSOR_X_0  
#define LWE397_SU_OUTER_SLI_SCISSOR_X_0                   (0x365)
#define LWE397_SU_OUTER_SLI_SCISSOR_X_0_MAX                 12:0

#define LWE397_SU_OUTER_SLI_SCISSOR_X_0_MIN                 27:16


// Register LWE397_SU_OUTER_SLI_SCISSOR_Y_0  
#define LWE397_SU_OUTER_SLI_SCISSOR_Y_0                   (0x366)
#define LWE397_SU_OUTER_SLI_SCISSOR_Y_0_MAX                 12:0

#define LWE397_SU_OUTER_SLI_SCISSOR_Y_0_MIN                 27:16

/*
#define QRAST_SCOREBOARD_DEPTH  64
#define QRAST_SCOREBOARD_WIDTH  64
#define QRAST_QUAD_COUNT_WIDTH  22
#define QRAST_NUM_EDGES_PER_TRIANGLE    4
#define QRAST_NUM_SAMPLES_PER_PIXEL     5
#define QRAST_NUM_VIRT_SAMPLES_PER_PIXEL        4
#define QRAST_NUM_PIXEL_ROWS_PER_QUAD   2
#define QRAST_NUM_PIXEL_COLS_PER_QUAD   2
#define QRAST_NUM_PIXEL_ROWS_PER_FINE_STAMP     3
#define QRAST_NUM_PIXEL_COLS_PER_FINE_STAMP     3
#define QRAST_FDC_CACHE_LINE_LEN        8
#define QRAST_SWATH_HEIGHT_IN_PIXELS    16
#define QRAST_TILE_HEIGHT_IN_PIXELS     18
#define QRAST_SCOREBOARD_TILE_X_SIZE    16
#define QRAST_SCOREBOARD_TILE_Y_SIZE    16
#define QRAST_SCOREBOARD_SIZE   2048
#define QRAST_SCOREBOARD_CACHEA_SIZE    7
#define QRAST_SCOREBOARD_CACHEB_SIZE    2
#define QRAST_NUM_QRAST_FDC_READ_PORTS  2
#define QRAST_NUM_QRAST_FDC_WRITE_PORTS 2
#define QRAST_ZSER_FIFO_DEPTH   64
#define QRAST_REAL_SAMP_X       0
#define QRAST_REAL_SAMP_Y       0
#define QRAST_VIRT_A_SAMP_X     6
#define QRAST_VIRT_A_SAMP_Y     2
#define QRAST_VIRT_B_SAMP_X     14
#define QRAST_VIRT_B_SAMP_Y     6
#define QRAST_VIRT_C_SAMP_X     2
#define QRAST_VIRT_C_SAMP_Y     10
#define QRAST_VIRT_D_SAMP_X     10
#define QRAST_VIRT_D_SAMP_Y     14
*/ 
 
// Register LWE397_QR_S_TEST_0  
#define LWE397_QR_S_TEST_0                        (0x400)
#define LWE397_QR_S_TEST_0_S_MASK                   7:0

#define LWE397_QR_S_TEST_0_S_FUNC                   10:8
#define LWE397_QR_S_TEST_0_S_FUNC_NEVER                   (0)
#define LWE397_QR_S_TEST_0_S_FUNC_LESS                    (1)
#define LWE397_QR_S_TEST_0_S_FUNC_EQUAL                   (2)
#define LWE397_QR_S_TEST_0_S_FUNC_LEQUAL                  (3)
#define LWE397_QR_S_TEST_0_S_FUNC_GREATER                 (4)
#define LWE397_QR_S_TEST_0_S_FUNC_NOTEQUAL                        (5)
#define LWE397_QR_S_TEST_0_S_FUNC_GEQUAL                  (6)
#define LWE397_QR_S_TEST_0_S_FUNC_ALWAYS                  (7)


// Register LWE397_QR_S_TEST  
#define LWE397_QR_S_TEST                  (0x400)
#define LWE397_QR_S_TEST_S_MASK                     7:0

#define LWE397_QR_S_TEST_S_FUNC                     10:8
#define LWE397_QR_S_TEST_S_FUNC_NEVER                     (0)
#define LWE397_QR_S_TEST_S_FUNC_LESS                      (1)
#define LWE397_QR_S_TEST_S_FUNC_EQUAL                     (2)
#define LWE397_QR_S_TEST_S_FUNC_LEQUAL                    (3)
#define LWE397_QR_S_TEST_S_FUNC_GREATER                   (4)
#define LWE397_QR_S_TEST_S_FUNC_NOTEQUAL                  (5)
#define LWE397_QR_S_TEST_S_FUNC_GEQUAL                    (6)
#define LWE397_QR_S_TEST_S_FUNC_ALWAYS                    (7)


// Register LWE397_QR_S_TEST_1  
#define LWE397_QR_S_TEST_1                        (0x401)
#define LWE397_QR_S_TEST_1_S_MASK                   7:0

#define LWE397_QR_S_TEST_1_S_FUNC                   10:8
#define LWE397_QR_S_TEST_1_S_FUNC_NEVER                   (0)
#define LWE397_QR_S_TEST_1_S_FUNC_LESS                    (1)
#define LWE397_QR_S_TEST_1_S_FUNC_EQUAL                   (2)
#define LWE397_QR_S_TEST_1_S_FUNC_LEQUAL                  (3)
#define LWE397_QR_S_TEST_1_S_FUNC_GREATER                 (4)
#define LWE397_QR_S_TEST_1_S_FUNC_NOTEQUAL                        (5)
#define LWE397_QR_S_TEST_1_S_FUNC_GEQUAL                  (6)
#define LWE397_QR_S_TEST_1_S_FUNC_ALWAYS                  (7)


// Register LWE397_QR_S_CTRL_0  
#define LWE397_QR_S_CTRL_0                        (0x402)
#define LWE397_QR_S_CTRL_0_COVERAGE_MERGE                   1:0
#define LWE397_QR_S_CTRL_0_COVERAGE_MERGE_NONE                    (0)
#define LWE397_QR_S_CTRL_0_COVERAGE_MERGE_OR                      (1)
#define LWE397_QR_S_CTRL_0_COVERAGE_MERGE_XOR                     (2)
#define LWE397_QR_S_CTRL_0_COVERAGE_MERGE_OR_NOT                  (3)

#define LWE397_QR_S_CTRL_0_S_SURF_PTR                       4:2

#define LWE397_QR_S_CTRL_0_S_ENABLE                 5:5
#define LWE397_QR_S_CTRL_0_S_ENABLE_DISABLE                       (0)
#define LWE397_QR_S_CTRL_0_S_ENABLE_ENABLE                        (1)

#define LWE397_QR_S_CTRL_0_QRAST_FB_WRITE                   6:6
#define LWE397_QR_S_CTRL_0_QRAST_FB_WRITE_DISABLE                 (0)
#define LWE397_QR_S_CTRL_0_QRAST_FB_WRITE_ENABLE                  (1)


// Register LWE397_QR_Z_TEST_0  
#define LWE397_QR_Z_TEST_0                        (0x403)
#define LWE397_QR_Z_TEST_0_Z_SURF_PTR                       2:0

#define LWE397_QR_Z_TEST_0_Z_ENABLE                 3:3
#define LWE397_QR_Z_TEST_0_Z_ENABLE_DISABLE                       (0)
#define LWE397_QR_Z_TEST_0_Z_ENABLE_ENABLE                        (1)

#define LWE397_QR_Z_TEST_0_Z_FUNC                   7:4
#define LWE397_QR_Z_TEST_0_Z_FUNC_NEVER                   (0)
#define LWE397_QR_Z_TEST_0_Z_FUNC_LESS                    (1)
#define LWE397_QR_Z_TEST_0_Z_FUNC_EQUAL                   (2)
#define LWE397_QR_Z_TEST_0_Z_FUNC_LEQUAL                  (3)
#define LWE397_QR_Z_TEST_0_Z_FUNC_GREATER                 (4)
#define LWE397_QR_Z_TEST_0_Z_FUNC_NOTEQUAL                        (5)
#define LWE397_QR_Z_TEST_0_Z_FUNC_GEQUAL                  (6)
#define LWE397_QR_Z_TEST_0_Z_FUNC_ALWAYS                  (7)

#define LWE397_QR_Z_TEST_0_QRAST_FB_WRITE                   8:8
#define LWE397_QR_Z_TEST_0_QRAST_FB_WRITE_DISABLE                 (0)
#define LWE397_QR_Z_TEST_0_QRAST_FB_WRITE_ENABLE                  (1)

#define LWE397_QR_Z_TEST_0_Z_CLAMP                  9:9
#define LWE397_QR_Z_TEST_0_Z_CLAMP_CLAMP                  (0)
#define LWE397_QR_Z_TEST_0_Z_CLAMP_KILL                   (1)

#define LWE397_QR_Z_TEST_0_Z_CLAMP_OVERRIDE                 10:10
#define LWE397_QR_Z_TEST_0_Z_CLAMP_OVERRIDE_DISABLE                       (0)
#define LWE397_QR_Z_TEST_0_Z_CLAMP_OVERRIDE_ENABLE                        (1)


// Register LWE397_QR_Z_MIN_0  
#define LWE397_QR_Z_MIN_0                 (0x404)
#define LWE397_QR_Z_MIN_0_VALUE                     19:0


// Register LWE397_QR_Z_MAX_0  
#define LWE397_QR_Z_MAX_0                 (0x405)
#define LWE397_QR_Z_MAX_0_VALUE                     19:0


// Register LWE397_QR_RAST_OPERATION_0  
#define LWE397_QR_RAST_OPERATION_0                        (0x406)
#define LWE397_QR_RAST_OPERATION_0_SWATH_WIDTH                      1:0
#define LWE397_QR_RAST_OPERATION_0_SWATH_WIDTH_PIX_8                      (0)
#define LWE397_QR_RAST_OPERATION_0_SWATH_WIDTH_PIX_16                     (1)

#define LWE397_QR_RAST_OPERATION_0_V_DIRECTION                      2:2
#define LWE397_QR_RAST_OPERATION_0_V_DIRECTION_DOWN                       (0)
#define LWE397_QR_RAST_OPERATION_0_V_DIRECTION_ALTERNATE                  (1)


// Register LWE397_QR_RAST_SCISSOR_SNAP_0  
#define LWE397_QR_RAST_SCISSOR_SNAP_0                     (0x407)
#define LWE397_QR_RAST_SCISSOR_SNAP_0_RAST_SCISSOR_SNAP                     1:0
#define LWE397_QR_RAST_SCISSOR_SNAP_0_RAST_SCISSOR_SNAP_NOP                       (0)
#define LWE397_QR_RAST_SCISSOR_SNAP_0_RAST_SCISSOR_SNAP_SAMPLE                    (1)
#define LWE397_QR_RAST_SCISSOR_SNAP_0_RAST_SCISSOR_SNAP_STALL                     (2)

#define LWE397_QR_RAST_SCISSOR_SNAP_0_RAST_SCISSOR_UPDATE                   2:2

#define LWE397_QR_RAST_SCISSOR_SNAP_0_RAST_SCISSOR_UPDATE_WRITE                     3:3

#define LWE397_QR_RAST_SCISSOR_SNAP_0_BBOX_ACLWMULATE                       4:4
#define LWE397_QR_RAST_SCISSOR_SNAP_0_BBOX_ACLWMULATE_DISABLE                     (0)
#define LWE397_QR_RAST_SCISSOR_SNAP_0_BBOX_ACLWMULATE_ENABLE                      (1)

#define LWE397_QR_RAST_SCISSOR_SNAP_0_RAST_SCISSOR_ENABLE                   5:5
#define LWE397_QR_RAST_SCISSOR_SNAP_0_RAST_SCISSOR_ENABLE_DISABLE                 (0)
#define LWE397_QR_RAST_SCISSOR_SNAP_0_RAST_SCISSOR_ENABLE_ENABLE                  (1)


// Register LWE397_QR_RAST_SCISSOR_MIN_0  
#define LWE397_QR_RAST_SCISSOR_MIN_0                      (0x408)
#define LWE397_QR_RAST_SCISSOR_MIN_0_X                      12:0

#define LWE397_QR_RAST_SCISSOR_MIN_0_Y                      28:16


// Register LWE397_QR_RAST_SCISSOR_MAX_0  
#define LWE397_QR_RAST_SCISSOR_MAX_0                      (0x409)
#define LWE397_QR_RAST_SCISSOR_MAX_0_X                      12:0

#define LWE397_QR_RAST_SCISSOR_MAX_0_Y                      28:16


// Register LWE397_QR_RAST_BBOX_MIN_0  
#define LWE397_QR_RAST_BBOX_MIN_0                 (0x40a)
#define LWE397_QR_RAST_BBOX_MIN_0_X                 12:0

#define LWE397_QR_RAST_BBOX_MIN_0_Y                 28:16


// Register LWE397_QR_RAST_BBOX_MAX_0  
#define LWE397_QR_RAST_BBOX_MAX_0                 (0x40b)
#define LWE397_QR_RAST_BBOX_MAX_0_X                 12:0

#define LWE397_QR_RAST_BBOX_MAX_0_Y                 28:16


// Register LWE397_QR_SB_OPERATION_0  
#define LWE397_QR_SB_OPERATION_0                  (0x40c)
#define LWE397_QR_SB_OPERATION_0_GRANULARITY                        1:0
#define LWE397_QR_SB_OPERATION_0_GRANULARITY_OFF                  (0)
#define LWE397_QR_SB_OPERATION_0_GRANULARITY_PER_SAMPLE                   (1)
#define LWE397_QR_SB_OPERATION_0_GRANULARITY_PER_PIXEL                    (2)
#define LWE397_QR_SB_OPERATION_0_GRANULARITY_PSEUDO                       (3)

#define LWE397_QR_SB_OPERATION_0_INDEX_MODE                 3:2
#define LWE397_QR_SB_OPERATION_0_INDEX_MODE_TABLE                 (0)
#define LWE397_QR_SB_OPERATION_0_INDEX_MODE_FORMULA                       (1)
#define LWE397_QR_SB_OPERATION_0_INDEX_MODE_RESERVED_1                    (2)
#define LWE397_QR_SB_OPERATION_0_INDEX_MODE_RESERVED_2                    (3)

#define LWE397_QR_SB_OPERATION_0_BLOCKING_MODE                      5:4
#define LWE397_QR_SB_OPERATION_0_BLOCKING_MODE_OFF                        (0)
#define LWE397_QR_SB_OPERATION_0_BLOCKING_MODE_PER_TRI_RAST                       (1)
#define LWE397_QR_SB_OPERATION_0_BLOCKING_MODE_PER_TRI_SHADER                     (2)

#define LWE397_QR_SB_OPERATION_0_STALL                      6:6

#define LWE397_QR_SB_OPERATION_0_CLEAR                      7:7

#define LWE397_QR_SB_OPERATION_0_CACHE                      8:8
#define LWE397_QR_SB_OPERATION_0_CACHE_ON                 (0)
#define LWE397_QR_SB_OPERATION_0_CACHE_OFF                        (1)

#define LWE397_QR_SB_OPERATION_0_RD_OPTZ_CYA                        9:9


// Register LWE397_QR_QRAST_CLKEN_OVERRIDE_0  
#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0                  (0x40d)
#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0_PIXTEST_CLKEN_OVR                  0:0
#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0_PIXTEST_CLKEN_OVR_CLK_GATED                      (0)
#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0_PIXTEST_CLKEN_OVR_CLK_ALWAYS_ON                  (1)

#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0_FDCIF_CLKEN_OVR                    1:1
#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0_FDCIF_CLKEN_OVR_CLK_GATED                        (0)
#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0_FDCIF_CLKEN_OVR_CLK_ALWAYS_ON                    (1)

#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0_TOP_CLKEN_OVR                      2:2
#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0_TOP_CLKEN_OVR_CLK_GATED                  (0)
#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0_TOP_CLKEN_OVR_CLK_ALWAYS_ON                      (1)

#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0_SCOREBOARD_CLKEN_OVR                       3:3
#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0_SCOREBOARD_CLKEN_OVR_CLK_GATED                   (0)
#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0_SCOREBOARD_CLKEN_OVR_CLK_ALWAYS_ON                       (1)

#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0_LATENCYFIFO_CLKEN_OVR                      4:4
#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0_LATENCYFIFO_CLKEN_OVR_CLK_GATED                  (0)
#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0_LATENCYFIFO_CLKEN_OVR_CLK_ALWAYS_ON                      (1)

#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0_ZSER_CLKEN_OVR                     5:5
#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0_ZSER_CLKEN_OVR_CLK_GATED                 (0)
#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0_ZSER_CLKEN_OVR_CLK_ALWAYS_ON                     (1)

#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0_REG_CLKEN_OVR                      6:6
#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0_REG_CLKEN_OVR_CLK_GATED                  (0)
#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0_REG_CLKEN_OVR_CLK_ALWAYS_ON                      (1)

#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0_SETUPIF_CLKEN_OVR                  7:7
#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0_SETUPIF_CLKEN_OVR_CLK_GATED                      (0)
#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0_SETUPIF_CLKEN_OVR_CLK_ALWAYS_ON                  (1)

#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0_TIDZSER_CLKEN_OVR                  8:8
#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0_TIDZSER_CLKEN_OVR_CLK_GATED                      (0)
#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0_TIDZSER_CLKEN_OVR_CLK_ALWAYS_ON                  (1)

#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0_PIPE_CLKEN_OVR                     9:9
#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0_PIPE_CLKEN_OVR_CLK_GATED                 (0)
#define LWE397_QR_QRAST_CLKEN_OVERRIDE_0_PIPE_CLKEN_OVR_CLK_ALWAYS_ON                     (1)


// Register LWE397_QR_VCAA_OPERATION_0  
#define LWE397_QR_VCAA_OPERATION_0                        (0x40e)
#define LWE397_QR_VCAA_OPERATION_0_QRAST_FB_WRITE                   1:0
#define LWE397_QR_VCAA_OPERATION_0_QRAST_FB_WRITE_DISABLE                 (0)
#define LWE397_QR_VCAA_OPERATION_0_QRAST_FB_WRITE_UNMERGE                 (1)
#define LWE397_QR_VCAA_OPERATION_0_QRAST_FB_WRITE_MERGE                   (2)
#define LWE397_QR_VCAA_OPERATION_0_QRAST_FB_WRITE_ONE                     (3)

#define LWE397_QR_VCAA_OPERATION_0_TRANSPARENCY                     2:2
#define LWE397_QR_VCAA_OPERATION_0_TRANSPARENCY_DISABLE                   (0)
#define LWE397_QR_VCAA_OPERATION_0_TRANSPARENCY_ENABLE                    (1)

#define LWE397_QR_VCAA_OPERATION_0_SILHOUETTE                       3:3
#define LWE397_QR_VCAA_OPERATION_0_SILHOUETTE_DISABLE                     (0)
#define LWE397_QR_VCAA_OPERATION_0_SILHOUETTE_ENABLE                      (1)

#define LWE397_QR_VCAA_OPERATION_0_VCAA_SURF_PTR                    6:4

#define LWE397_QR_VCAA_OPERATION_0_VCAA_ENABLE                      8:7
#define LWE397_QR_VCAA_OPERATION_0_VCAA_ENABLE_DISABLE                    (0)
#define LWE397_QR_VCAA_OPERATION_0_VCAA_ENABLE_ENABLE                     (1)
#define LWE397_QR_VCAA_OPERATION_0_VCAA_ENABLE_COMPUTE_ONLY                       (2)

#define LWE397_QR_VCAA_OPERATION_0_VIRTUAL_AS_REAL                  9:9
#define LWE397_QR_VCAA_OPERATION_0_VIRTUAL_AS_REAL_DISABLE                        (0)
#define LWE397_QR_VCAA_OPERATION_0_VIRTUAL_AS_REAL_ENABLE                 (1)


// Register LWE397_QR_OUTPUT_TO_SHADER_0  
#define LWE397_QR_OUTPUT_TO_SHADER_0                      (0x40f)
#define LWE397_QR_OUTPUT_TO_SHADER_0_VCAA_OUTPUT                    2:0
#define LWE397_QR_OUTPUT_TO_SHADER_0_VCAA_OUTPUT_DISABLE                  (0)
#define LWE397_QR_OUTPUT_TO_SHADER_0_VCAA_OUTPUT_MERGE                    (1)
#define LWE397_QR_OUTPUT_TO_SHADER_0_VCAA_OUTPUT_UNMERGE                  (2)
#define LWE397_QR_OUTPUT_TO_SHADER_0_VCAA_OUTPUT_SMEAR                    (3)
#define LWE397_QR_OUTPUT_TO_SHADER_0_VCAA_OUTPUT_STENCIL_COVERAGE                 (4)

#define LWE397_QR_OUTPUT_TO_SHADER_0_SWALLOW_QUADS                  4:3
#define LWE397_QR_OUTPUT_TO_SHADER_0_SWALLOW_QUADS_OFF                    (0)
#define LWE397_QR_OUTPUT_TO_SHADER_0_SWALLOW_QUADS_VIRTUAL                        (1)
#define LWE397_QR_OUTPUT_TO_SHADER_0_SWALLOW_QUADS_ALL                    (2)

#define LWE397_QR_OUTPUT_TO_SHADER_0_PASS_ONLY                      5:5
#define LWE397_QR_OUTPUT_TO_SHADER_0_PASS_ONLY_ENABLE                     (0)
#define LWE397_QR_OUTPUT_TO_SHADER_0_PASS_ONLY_DISABLE                    (1)

#define LWE397_QR_OUTPUT_TO_SHADER_0_SCOREBOARD_OUTPUT                      6:6
#define LWE397_QR_OUTPUT_TO_SHADER_0_SCOREBOARD_OUTPUT_DISABLE                    (0)
#define LWE397_QR_OUTPUT_TO_SHADER_0_SCOREBOARD_OUTPUT_ENABLE                     (1)

#define LWE397_QR_OUTPUT_TO_SHADER_0_TIMEOUT                        9:7

#define LWE397_QR_OUTPUT_TO_SHADER_0_PSEQ_FLUSH                     10:10
#define LWE397_QR_OUTPUT_TO_SHADER_0_PSEQ_FLUSH_PERFORMANCE                       (0)
#define LWE397_QR_OUTPUT_TO_SHADER_0_PSEQ_FLUSH_DETERMINISTIC                     (1)

#define LWE397_QR_OUTPUT_TO_SHADER_0_OVERLAP                        11:11
#define LWE397_QR_OUTPUT_TO_SHADER_0_OVERLAP_DISABLE                      (0)
#define LWE397_QR_OUTPUT_TO_SHADER_0_OVERLAP_ENABLE                       (1)


// Register LWE397_QR_QRAST_DEBUG_0  
#define LWE397_QR_QRAST_DEBUG_0                   (0x410)
#define LWE397_QR_QRAST_DEBUG_0_VALUE                       31:0


// Register LWE397_QR_QRAST_LIMITS_0  
#define LWE397_QR_QRAST_LIMITS_0                  (0x411)
#define LWE397_QR_QRAST_LIMITS_0_PIX_CNT_QRAST                      11:0

#define LWE397_QR_QRAST_LIMITS_0_PIX_CNT_SHD                        22:12


// Packet ROW_RAM
#define ROW_RAM_SIZE 251

#define ROW_RAM_SMP0_IN_ROW                     0

#define ROW_RAM_SMP1_IN_ROW                     0

#define ROW_RAM_SMP2_IN_ROW                     0

#define ROW_RAM_SMP3_IN_ROW                     0

#define ROW_RAM_SMP4_IN_ROW                     0

#define ROW_RAM_SMP5_IN_ROW                     0

#define ROW_RAM_SMP6_IN_ROW                     0

#define ROW_RAM_SMP7_IN_ROW                     0

#define ROW_RAM_SMP8_IN_ROW                     0

#define ROW_RAM_SMP9_IN_ROW                     0

#define ROW_RAM_SMP0_OUT_ROW                    0

#define ROW_RAM_SMP1_OUT_ROW                    0

#define ROW_RAM_SMP2_OUT_ROW                    0

#define ROW_RAM_SMP3_OUT_ROW                    0

#define ROW_RAM_SMP4_OUT_ROW                    0

#define ROW_RAM_SMP5_OUT_ROW                    0

#define ROW_RAM_SMP6_OUT_ROW                    0

#define ROW_RAM_SMP7_OUT_ROW                    0

#define ROW_RAM_SMP8_OUT_ROW                    0

#define ROW_RAM_SMP9_OUT_ROW                    0

#define ROW_RAM_SMP_COVERED_ROW                 0


// Register LWE397_QR_PIXEL_COUNT_CTRL_0  
#define LWE397_QR_PIXEL_COUNT_CTRL_0                      (0x412)
#define LWE397_QR_PIXEL_COUNT_CTRL_0_ENABLE                 0:0

#define LWE397_QR_PIXEL_COUNT_CTRL_0_COUNT_NON_CENTER                       1:1


// Register LWE397_QR_PIXEL_COUNT_0  
#define LWE397_QR_PIXEL_COUNT_0                   (0x413)
#define LWE397_QR_PIXEL_COUNT_0_VALUE                       31:0


// Register LWE397_PSEQ_FLUSH_0  
#define LWE397_PSEQ_FLUSH_0                       (0x500)
#define LWE397_PSEQ_FLUSH_0_FLUSH                   0:0


// Register LWE397_PSEQ_CTL_0  
#define LWE397_PSEQ_CTL_0                 (0x501)
#define LWE397_PSEQ_CTL_0_MERGE_SPAN_STARTS                 0:0

#define LWE397_PSEQ_CTL_0_MERGE_REGISTERS                   1:1

#define LWE397_PSEQ_CTL_0_REMOVE_KILLED_PIXELS                      2:2

#define LWE397_PSEQ_CTL_0_ALLOW_QID_COLLISIONS                      3:3

#define LWE397_PSEQ_CTL_0_MAX_OUT                   13:4

#define LWE397_PSEQ_CTL_0_MIN_OUT                   23:14


// Register LWE397_PSEQ_TIMEOUT_0  
#define LWE397_PSEQ_TIMEOUT_0                     (0x502)
#define LWE397_PSEQ_TIMEOUT_0_COUNT                 31:0


// Register LWE397_PSEQ_PC_0  
#define LWE397_PSEQ_PC_0                  (0x503)
#define LWE397_PSEQ_PC_0_PC                 3:0


// Register LWE397_PSEQ_COMMAND_0  
#define LWE397_PSEQ_COMMAND_0                     (0x520)
#define LWE397_PSEQ_COMMAND_0_DATA                  31:0


// Register LWE397_PSEQ_COMMAND  
#define LWE397_PSEQ_COMMAND                       (0x520)
#define LWE397_PSEQ_COMMAND_DATA                    31:0


// Register LWE397_PSEQ_COMMAND_1  
#define LWE397_PSEQ_COMMAND_1                     (0x521)
#define LWE397_PSEQ_COMMAND_1_DATA                  31:0


// Register LWE397_PSEQ_COMMAND_2  
#define LWE397_PSEQ_COMMAND_2                     (0x522)
#define LWE397_PSEQ_COMMAND_2_DATA                  31:0


// Register LWE397_PSEQ_COMMAND_3  
#define LWE397_PSEQ_COMMAND_3                     (0x523)
#define LWE397_PSEQ_COMMAND_3_DATA                  31:0


// Register LWE397_PSEQ_COMMAND_4  
#define LWE397_PSEQ_COMMAND_4                     (0x524)
#define LWE397_PSEQ_COMMAND_4_DATA                  31:0


// Register LWE397_PSEQ_COMMAND_5  
#define LWE397_PSEQ_COMMAND_5                     (0x525)
#define LWE397_PSEQ_COMMAND_5_DATA                  31:0


// Register LWE397_PSEQ_COMMAND_6  
#define LWE397_PSEQ_COMMAND_6                     (0x526)
#define LWE397_PSEQ_COMMAND_6_DATA                  31:0


// Register LWE397_PSEQ_COMMAND_7  
#define LWE397_PSEQ_COMMAND_7                     (0x527)
#define LWE397_PSEQ_COMMAND_7_DATA                  31:0


// Register LWE397_PSEQ_COMMAND_8  
#define LWE397_PSEQ_COMMAND_8                     (0x528)
#define LWE397_PSEQ_COMMAND_8_DATA                  31:0


// Register LWE397_PSEQ_COMMAND_9  
#define LWE397_PSEQ_COMMAND_9                     (0x529)
#define LWE397_PSEQ_COMMAND_9_DATA                  31:0


// Register LWE397_PSEQ_COMMAND_10  
#define LWE397_PSEQ_COMMAND_10                    (0x52a)
#define LWE397_PSEQ_COMMAND_10_DATA                 31:0


// Register LWE397_PSEQ_COMMAND_11  
#define LWE397_PSEQ_COMMAND_11                    (0x52b)
#define LWE397_PSEQ_COMMAND_11_DATA                 31:0


// Register LWE397_PSEQ_COMMAND_12  
#define LWE397_PSEQ_COMMAND_12                    (0x52c)
#define LWE397_PSEQ_COMMAND_12_DATA                 31:0


// Register LWE397_PSEQ_COMMAND_13  
#define LWE397_PSEQ_COMMAND_13                    (0x52d)
#define LWE397_PSEQ_COMMAND_13_DATA                 31:0


// Register LWE397_PSEQ_COMMAND_14  
#define LWE397_PSEQ_COMMAND_14                    (0x52e)
#define LWE397_PSEQ_COMMAND_14_DATA                 31:0


// Register LWE397_PSEQ_COMMAND_15  
#define LWE397_PSEQ_COMMAND_15                    (0x52f)
#define LWE397_PSEQ_COMMAND_15_DATA                 31:0


// Register LWE397_PSEQ_COMMAND_16  
#define LWE397_PSEQ_COMMAND_16                    (0x530)
#define LWE397_PSEQ_COMMAND_16_DATA                 31:0


// Register LWE397_PSEQ_COMMAND_17  
#define LWE397_PSEQ_COMMAND_17                    (0x531)
#define LWE397_PSEQ_COMMAND_17_DATA                 31:0


// Register LWE397_PSEQ_COMMAND_18  
#define LWE397_PSEQ_COMMAND_18                    (0x532)
#define LWE397_PSEQ_COMMAND_18_DATA                 31:0


// Register LWE397_PSEQ_COMMAND_19  
#define LWE397_PSEQ_COMMAND_19                    (0x533)
#define LWE397_PSEQ_COMMAND_19_DATA                 31:0


// Register LWE397_PSEQ_COMMAND_20  
#define LWE397_PSEQ_COMMAND_20                    (0x534)
#define LWE397_PSEQ_COMMAND_20_DATA                 31:0


// Register LWE397_PSEQ_COMMAND_21  
#define LWE397_PSEQ_COMMAND_21                    (0x535)
#define LWE397_PSEQ_COMMAND_21_DATA                 31:0


// Register LWE397_PSEQ_COMMAND_22  
#define LWE397_PSEQ_COMMAND_22                    (0x536)
#define LWE397_PSEQ_COMMAND_22_DATA                 31:0


// Register LWE397_PSEQ_COMMAND_23  
#define LWE397_PSEQ_COMMAND_23                    (0x537)
#define LWE397_PSEQ_COMMAND_23_DATA                 31:0


// Register LWE397_PSEQ_COMMAND_24  
#define LWE397_PSEQ_COMMAND_24                    (0x538)
#define LWE397_PSEQ_COMMAND_24_DATA                 31:0


// Register LWE397_PSEQ_COMMAND_25  
#define LWE397_PSEQ_COMMAND_25                    (0x539)
#define LWE397_PSEQ_COMMAND_25_DATA                 31:0


// Register LWE397_PSEQ_COMMAND_26  
#define LWE397_PSEQ_COMMAND_26                    (0x53a)
#define LWE397_PSEQ_COMMAND_26_DATA                 31:0


// Register LWE397_PSEQ_COMMAND_27  
#define LWE397_PSEQ_COMMAND_27                    (0x53b)
#define LWE397_PSEQ_COMMAND_27_DATA                 31:0


// Register LWE397_PSEQ_COMMAND_28  
#define LWE397_PSEQ_COMMAND_28                    (0x53c)
#define LWE397_PSEQ_COMMAND_28_DATA                 31:0


// Register LWE397_PSEQ_COMMAND_29  
#define LWE397_PSEQ_COMMAND_29                    (0x53d)
#define LWE397_PSEQ_COMMAND_29_DATA                 31:0


// Register LWE397_PSEQ_COMMAND_30  
#define LWE397_PSEQ_COMMAND_30                    (0x53e)
#define LWE397_PSEQ_COMMAND_30_DATA                 31:0


// Register LWE397_PSEQ_COMMAND_31  
#define LWE397_PSEQ_COMMAND_31                    (0x53f)
#define LWE397_PSEQ_COMMAND_31_DATA                 31:0


// Packet LWE397_PSEQ_CMD
#define LWE397_PSEQ_CMD_SIZE 32

#define LWE397_PSEQ_CMD_OPCODE_ROW                        0
#define LWE397_PSEQ_CMD_OPCODE_GATHER                     (0)
#define LWE397_PSEQ_CMD_OPCODE_EXELWTE                    (1)
#define LWE397_PSEQ_CMD_OPCODE_BRANCH                     (2)
#define LWE397_PSEQ_CMD_OPCODE_IMM                        (3)
#define LWE397_PSEQ_CMD_OPCODE_ST                 (4)

#define LWE397_PSEQ_CMD_ARGS0_ROW                 0

#define LWE397_PSEQ_CMD_ARGS1_ROW                 1


// Packet LWE397_PSEQ_GATHER
#define LWE397_PSEQ_GATHER_SIZE 32

#define LWE397_PSEQ_GATHER_OPCODE_ROW                     0
#define LWE397_PSEQ_GATHER_OPCODE_GATHER                  (0)

#define LWE397_PSEQ_GATHER_STOP_ROW                       0

#define LWE397_PSEQ_GATHER_OFFSET_ROW                     0

#define LWE397_PSEQ_GATHER_INSERT_ROW                     0
#define LWE397_PSEQ_GATHER_INSERT_DISABLE                 (0)
#define LWE397_PSEQ_GATHER_INSERT_ENABLE                  (1)

#define LWE397_PSEQ_GATHER_TYPE_ROW                       0
#define LWE397_PSEQ_GATHER_TYPE_NONINCR                   (0)
#define LWE397_PSEQ_GATHER_TYPE_INCR                      (1)

#define LWE397_PSEQ_GATHER_CACHE_PERSISTENT_ROW                   0

#define LWE397_PSEQ_GATHER_COUNT_ROW                      0

#define LWE397_PSEQ_GATHER_ADDRESS_ROW                    1


// Packet LWE397_PSEQ_EXELWTE
#define LWE397_PSEQ_EXELWTE_SIZE 32

#define LWE397_PSEQ_EXELWTE_OPCODE_ROW                    0
#define LWE397_PSEQ_EXELWTE_OPCODE_EXELWTE                        (1)

#define LWE397_PSEQ_EXELWTE_UNUSED0_ROW                   0

#define LWE397_PSEQ_EXELWTE_NEXT_START_ROW                        0

#define LWE397_PSEQ_EXELWTE_DEST_ROW                      0
#define LWE397_PSEQ_EXELWTE_DEST_RECIRC                   (0)
#define LWE397_PSEQ_EXELWTE_DEST_SPILL                    (1)
#define LWE397_PSEQ_EXELWTE_DEST_DONE                     (2)
#define LWE397_PSEQ_EXELWTE_DEST_STOP                     (3)

#define LWE397_PSEQ_EXELWTE_START_ROW                     0

#define LWE397_PSEQ_EXELWTE_COUNT_ROW                     0

#define LWE397_PSEQ_EXELWTE_UNUSED1_ROW                   1


// Packet LWE397_PSEQ_BRANCH
#define LWE397_PSEQ_BRANCH_SIZE 32

#define LWE397_PSEQ_BRANCH_OPCODE_ROW                     0
#define LWE397_PSEQ_BRANCH_OPCODE_BRANCH                  (2)

#define LWE397_PSEQ_BRANCH_COND_ROW                       0
#define LWE397_PSEQ_BRANCH_COND_ALWAYS                    (0)
#define LWE397_PSEQ_BRANCH_COND_NEVER                     (1)
#define LWE397_PSEQ_BRANCH_COND_AND                       (2)
#define LWE397_PSEQ_BRANCH_COND_NAND                      (3)
#define LWE397_PSEQ_BRANCH_COND_OR                        (4)
#define LWE397_PSEQ_BRANCH_COND_NOR                       (5)

#define LWE397_PSEQ_BRANCH_DEST_ROW                       0

#define LWE397_PSEQ_BRANCH_UNUSED1_ROW                    1


// Packet LWE397_PSEQ_IMM
#define LWE397_PSEQ_IMM_SIZE 32

#define LWE397_PSEQ_IMM_OPCODE_ROW                        0
#define LWE397_PSEQ_IMM_OPCODE_IMM                        (3)

#define LWE397_PSEQ_IMM_OFFSET_ROW                        0

#define LWE397_PSEQ_IMM_UNUSED_ROW                        0

#define LWE397_PSEQ_IMM_IMMDATA_ROW                       1


// Packet LWE397_PSEQ_ST
#define LWE397_PSEQ_ST_SIZE 32

#define LWE397_PSEQ_ST_OPCODE_ROW                 0
#define LWE397_PSEQ_ST_OPCODE_ST                  (4)

#define LWE397_PSEQ_ST_OFFSET_ROW                 0

#define LWE397_PSEQ_ST_SURF_ROW                   0

#define LWE397_PSEQ_ST_TYPE_ROW                   0
#define LWE397_PSEQ_ST_TYPE_NONINCR                       (0)
#define LWE397_PSEQ_ST_TYPE_INCR                  (1)

#define LWE397_PSEQ_ST_COUNT_ROW                  0

#define LWE397_PSEQ_ST_UNUSED1_ROW                        1


// Register LWE397_PSEQ_INST_OFFSET_0  
#define LWE397_PSEQ_INST_OFFSET_0                 (0x540)
#define LWE397_PSEQ_INST_OFFSET_0_INDEX                     5:0


// Register LWE397_PSEQ_INST_DATA_0  
#define LWE397_PSEQ_INST_DATA_0                   (0x541)
#define LWE397_PSEQ_INST_DATA_0_OP                  24:23
#define LWE397_PSEQ_INST_DATA_0_OP_NOP                    (0)
#define LWE397_PSEQ_INST_DATA_0_OP_LD                     (1)
#define LWE397_PSEQ_INST_DATA_0_OP_LD_R20                 (2)
#define LWE397_PSEQ_INST_DATA_0_OP_LD_R80                 (3)

#define LWE397_PSEQ_INST_DATA_0_CACHE_PERSISTENT                    22:22

#define LWE397_PSEQ_INST_DATA_0_READ_KILLED                 21:21
#define LWE397_PSEQ_INST_DATA_0_READ_KILLED_DISABLE                       (0)
#define LWE397_PSEQ_INST_DATA_0_READ_KILLED_ENABLE                        (1)

#define LWE397_PSEQ_INST_DATA_0_READ_NON_CENTER                     20:20
#define LWE397_PSEQ_INST_DATA_0_READ_NON_CENTER_DISABLE                   (0)
#define LWE397_PSEQ_INST_DATA_0_READ_NON_CENTER_ENABLE                    (1)

#define LWE397_PSEQ_INST_DATA_0_SURF                        19:16

#define LWE397_PSEQ_INST_DATA_0_ARGS                        15:0

/*
// Packet LD_ARGS
#define LD_ARGS_SIZE 16

#define LD_ARGS_REG_ROW                 0
#define LD_ARGS_REG_R0                  (0)
#define LD_ARGS_REG_R1                  (1)
#define LD_ARGS_REG_R2                  (2)
#define LD_ARGS_REG_R3                  (3)

#define LD_ARGS_MOD_ROW                 0
#define LD_ARGS_MOD_L                   (0)
#define LD_ARGS_MOD_H                   (1)

#define LD_ARGS_CLW_ROW                 0
#define LD_ARGS_CLW_DISABLE                     (0)
#define LD_ARGS_CLW_ENABLE                      (1)

#define LD_ARGS_RAW_ROW                 0
#define LD_ARGS_RAW_DISABLE                     (0)
#define LD_ARGS_RAW_ENABLE                      (1)

#define LD_ARGS_RESERVED_ROW                    0


// Packet LD_R20_ARGS
#define LD_R20_ARGS_SIZE 16

#define LD_R20_ARGS_REG_ROW                     0
#define LD_R20_ARGS_REG_R0                      (0)
#define LD_R20_ARGS_REG_R1                      (1)
#define LD_R20_ARGS_REG_R2                      (2)
#define LD_R20_ARGS_REG_R3                      (3)

#define LD_R20_ARGS_READ_MASK_ROW                       0

#define LD_R20_ARGS_OFFSET_IMM_ROW                      0

#define LD_R20_ARGS_OFFSET_REG_ROW                      0
#define LD_R20_ARGS_OFFSET_REG_R0                       (0)
#define LD_R20_ARGS_OFFSET_REG_R1                       (1)
#define LD_R20_ARGS_OFFSET_REG_R2                       (2)
#define LD_R20_ARGS_OFFSET_REG_R3                       (3)

#define LD_R20_ARGS_OFFSET_MOD_ROW                      0
#define LD_R20_ARGS_OFFSET_MOD_L                        (0)
#define LD_R20_ARGS_OFFSET_MOD_H                        (1)

#define LD_R20_ARGS_OFFSET_REG_EN_ROW                   0
#define LD_R20_ARGS_OFFSET_REG_EN_DISABLE                       (0)
#define LD_R20_ARGS_OFFSET_REG_EN_ENABLE                        (1)

#define LD_R20_ARGS_OFFSET_TYPE_ROW                     0
#define LD_R20_ARGS_OFFSET_TYPE_LOCAL                   (0)
#define LD_R20_ARGS_OFFSET_TYPE_GLOBAL                  (1)


// Packet LD_R80_ARGS
#define LD_R80_ARGS_SIZE 16

#define LD_R80_ARGS_READ_MASK_ROW                       0

#define LD_R80_ARGS_OFFSET_IMM_ROW                      0

#define LD_R80_ARGS_OFFSET_REG_ROW                      0
#define LD_R80_ARGS_OFFSET_REG_R0                       (0)
#define LD_R80_ARGS_OFFSET_REG_R1                       (1)
#define LD_R80_ARGS_OFFSET_REG_R2                       (2)
#define LD_R80_ARGS_OFFSET_REG_R3                       (3)

#define LD_R80_ARGS_OFFSET_MOD_ROW                      0
#define LD_R80_ARGS_OFFSET_MOD_L                        (0)
#define LD_R80_ARGS_OFFSET_MOD_H                        (1)

#define LD_R80_ARGS_OFFSET_REG_EN_ROW                   0
#define LD_R80_ARGS_OFFSET_REG_EN_DISABLE                       (0)
#define LD_R80_ARGS_OFFSET_REG_EN_ENABLE                        (1)

#define LD_R80_ARGS_OFFSET_TYPE_ROW                     0
#define LD_R80_ARGS_OFFSET_TYPE_LOCAL                   (0)
#define LD_R80_ARGS_OFFSET_TYPE_GLOBAL                  (1)
*/

// Register LWE397_PSEQ_DBG_X_0  
#define LWE397_PSEQ_DBG_X_0                       (0x542)
#define LWE397_PSEQ_DBG_X_0_START                   21:11

#define LWE397_PSEQ_DBG_X_0_END                     10:0


// Register LWE397_PSEQ_DBG_Y_0  
#define LWE397_PSEQ_DBG_Y_0                       (0x543)
#define LWE397_PSEQ_DBG_Y_0_START                   21:11

#define LWE397_PSEQ_DBG_Y_0_END                     10:0


// Register LWE397_PSEQ_DBG_CTL_0  
#define LWE397_PSEQ_DBG_CTL_0                     (0x544)
#define LWE397_PSEQ_DBG_CTL_0_X_EN                  0:0
#define LWE397_PSEQ_DBG_CTL_0_X_EN_DISABLE                        (0)
#define LWE397_PSEQ_DBG_CTL_0_X_EN_ENABLE                 (1)

#define LWE397_PSEQ_DBG_CTL_0_Y_EN                  1:1
#define LWE397_PSEQ_DBG_CTL_0_Y_EN_DISABLE                        (0)
#define LWE397_PSEQ_DBG_CTL_0_Y_EN_ENABLE                 (1)

#define LWE397_PSEQ_DBG_CTL_0_SEQ_EN                        2:2
#define LWE397_PSEQ_DBG_CTL_0_SEQ_EN_DISABLE                      (0)
#define LWE397_PSEQ_DBG_CTL_0_SEQ_EN_ENABLE                       (1)

#define LWE397_PSEQ_DBG_CTL_0_COMMAND_EN                    3:3
#define LWE397_PSEQ_DBG_CTL_0_COMMAND_EN_DISABLE                  (0)
#define LWE397_PSEQ_DBG_CTL_0_COMMAND_EN_ENABLE                   (1)

#define LWE397_PSEQ_DBG_CTL_0_SEQ_START                     9:4

#define LWE397_PSEQ_DBG_CTL_0_SEQ_END                       17:12

#define LWE397_PSEQ_DBG_CTL_0_COMMAND_PC                    27:24


// Register LWE397_PSEQ_QUAD_ID_0  
#define LWE397_PSEQ_QUAD_ID_0                     (0x545)
#define LWE397_PSEQ_QUAD_ID_0_INDEX                 7:0


// Register LWE397_PSEQ_DWR_IF_STATE_0  
#define LWE397_PSEQ_DWR_IF_STATE_0                        (0x546)
#define LWE397_PSEQ_DWR_IF_STATE_0_START                    5:0

#define LWE397_PSEQ_DWR_IF_STATE_0_COUNT                    12:6

#define LWE397_PSEQ_DWR_IF_STATE_0_NEXT                     18:13

#define LWE397_PSEQ_DWR_IF_STATE_0_NOT_LAST_EXE                     19:19


// Register LWE397_AT_REMAP_OFFSET_0  
#define LWE397_AT_REMAP_OFFSET_0                  (0x600)
#define LWE397_AT_REMAP_OFFSET_0_INDEX                      5:0

#define LWE397_AT_REMAP_OFFSET_0_BASE                       11:6


// Register LWE397_AT_REMAP_DATA_0  
#define LWE397_AT_REMAP_DATA_0                    (0x601)
#define LWE397_AT_REMAP_DATA_0_COUNT                        1:0

#define LWE397_AT_REMAP_DATA_0_OFFSET                       7:2


// Register LWE397_AT_REMAP_DATA_4X_0  
#define LWE397_AT_REMAP_DATA_4X_0                 (0x602)
#define LWE397_AT_REMAP_DATA_4X_0_COUNT0                    1:0

#define LWE397_AT_REMAP_DATA_4X_0_OFFSET0                   7:2

#define LWE397_AT_REMAP_DATA_4X_0_COUNT1                    9:8

#define LWE397_AT_REMAP_DATA_4X_0_OFFSET1                   15:10

#define LWE397_AT_REMAP_DATA_4X_0_COUNT2                    17:16

#define LWE397_AT_REMAP_DATA_4X_0_OFFSET2                   23:18

#define LWE397_AT_REMAP_DATA_4X_0_COUNT3                    25:24

#define LWE397_AT_REMAP_DATA_4X_0_OFFSET3                   31:26


// Register LWE397_AT_INST_OFFSET_0  
#define LWE397_AT_INST_OFFSET_0                   (0x603)
#define LWE397_AT_INST_OFFSET_0_INDEX                       6:0


// Packet LWE397_AT_FUNC
#define LWE397_AT_FUNC_SIZE 7

#define LWE397_AT_FUNC_SRC_DST_ROW                        0
#define LWE397_AT_FUNC_SRC_DST_R0                 (0)
#define LWE397_AT_FUNC_SRC_DST_R1                 (1)
#define LWE397_AT_FUNC_SRC_DST_R2                 (2)
#define LWE397_AT_FUNC_SRC_DST_R3                 (3)
#define LWE397_AT_FUNC_SRC_DST_W                  (4)

#define LWE397_AT_FUNC_FUNC_ROW                   0
#define LWE397_AT_FUNC_FUNC_NOP                   (0)
#define LWE397_AT_FUNC_FUNC_RCP                   (1)
#define LWE397_AT_FUNC_FUNC_RSQRT                 (2)
#define LWE397_AT_FUNC_FUNC_LOG2                  (3)
#define LWE397_AT_FUNC_FUNC_EXP2                  (4)
#define LWE397_AT_FUNC_FUNC_SQRT                  (5)
#define LWE397_AT_FUNC_FUNC_SIN                   (6)
#define LWE397_AT_FUNC_FUNC_COS                   (7)
#define LWE397_AT_FUNC_FUNC_FRC                   (8)
#define LWE397_AT_FUNC_FUNC_EXP2RRO                       (9)
#define LWE397_AT_FUNC_FUNC_SINRRO                        (10)
#define LWE397_AT_FUNC_FUNC_COSRRO                        (11)


// Packet LWE397_AT_MUL
#define LWE397_AT_MUL_SIZE 11

#define LWE397_AT_MUL_DST_ROW                     0
#define LWE397_AT_MUL_DST_NOP                     (0)
#define LWE397_AT_MUL_DST_PER                     (1)
#define LWE397_AT_MUL_DST_R0                      (4)
#define LWE397_AT_MUL_DST_R1                      (5)
#define LWE397_AT_MUL_DST_R2                      (6)
#define LWE397_AT_MUL_DST_R3                      (7)

#define LWE397_AT_MUL_SRC1_ROW                    0
#define LWE397_AT_MUL_SRC1_R0                     (0)
#define LWE397_AT_MUL_SRC1_R1                     (1)
#define LWE397_AT_MUL_SRC1_R2                     (2)
#define LWE397_AT_MUL_SRC1_R3                     (3)
#define LWE397_AT_MUL_SRC1_C0                     (4)
#define LWE397_AT_MUL_SRC1_C1                     (5)
#define LWE397_AT_MUL_SRC1_C2                     (6)
#define LWE397_AT_MUL_SRC1_C3                     (7)
#define LWE397_AT_MUL_SRC1_I0                     (8)
#define LWE397_AT_MUL_SRC1_I1                     (9)
#define LWE397_AT_MUL_SRC1_W                      (10)
#define LWE397_AT_MUL_SRC1_ALPHA                  (11)
#define LWE397_AT_MUL_SRC1_BETA                   (12)
#define LWE397_AT_MUL_SRC1_ONE                    (13)

#define LWE397_AT_MUL_SRC0_ROW                    0
#define LWE397_AT_MUL_SRC0_R0                     (0)
#define LWE397_AT_MUL_SRC0_R1                     (1)
#define LWE397_AT_MUL_SRC0_R2                     (2)
#define LWE397_AT_MUL_SRC0_R3                     (3)
#define LWE397_AT_MUL_SRC0_C0                     (4)
#define LWE397_AT_MUL_SRC0_C1                     (5)
#define LWE397_AT_MUL_SRC0_C2                     (6)
#define LWE397_AT_MUL_SRC0_C3                     (7)
#define LWE397_AT_MUL_SRC0_I0                     (8)
#define LWE397_AT_MUL_SRC0_I1                     (9)
#define LWE397_AT_MUL_SRC0_W                      (10)
#define LWE397_AT_MUL_SRC0_ALPHA                  (11)
#define LWE397_AT_MUL_SRC0_BETA                   (12)
#define LWE397_AT_MUL_SRC0_ONE                    (13)


// Packet LWE397_AT_IPA
#define LWE397_AT_IPA_SIZE 7

#define LWE397_AT_IPA_ENTRY_ROW                   0

#define LWE397_AT_IPA_PREC_ROW                    0
#define LWE397_AT_IPA_PREC_NOP                    (0)
#define LWE397_AT_IPA_PREC_ZP                     (1)
#define LWE397_AT_IPA_PREC_HPS                    (2)
#define LWE397_AT_IPA_PREC_HPC                    (3)
#define LWE397_AT_IPA_PREC_LPS_LPS                        (4)
#define LWE397_AT_IPA_PREC_LPS_LPC                        (5)
#define LWE397_AT_IPA_PREC_LPC_LPS                        (6)
#define LWE397_AT_IPA_PREC_LPC_LPC                        (7)


// Register LWE397_AT_INST_DATA_0  
#define LWE397_AT_INST_DATA_0                     (0x604)
#define LWE397_AT_INST_DATA_0_DATA0                 31:0

#define LWE397_AT_INST_DATA_0_IMM1H                 31:22

#define LWE397_AT_INST_DATA_0_LWBE                  29:29
#define LWE397_AT_INST_DATA_0_LWBE_NOP                    (0)
#define LWE397_AT_INST_DATA_0_LWBE_R0                     (1)

#define LWE397_AT_INST_DATA_0_FUNC                  28:22

#define LWE397_AT_INST_DATA_0_MUL1                  21:11

#define LWE397_AT_INST_DATA_0_MUL0                  10:0

#define LWE397_AT_INST_DATA_0_DATA1                 29:0

#define LWE397_AT_INST_DATA_0_IMM1L                 29:20

#define LWE397_AT_INST_DATA_0_IMM0                  19:0

#define LWE397_AT_INST_DATA_0_IPA3                  27:21

#define LWE397_AT_INST_DATA_0_IPA2                  20:14

#define LWE397_AT_INST_DATA_0_IPA1                  13:7

#define LWE397_AT_INST_DATA_0_IPA0                  6:0


// Register LWE397_AT_CONSTANT0_0  
#define LWE397_AT_CONSTANT0_0                     (0x608)
#define LWE397_AT_CONSTANT0_0_VAL                   19:0


// Register LWE397_AT_CONSTANT0  
#define LWE397_AT_CONSTANT0                       (0x608)
#define LWE397_AT_CONSTANT0_VAL                     19:0


// Register LWE397_AT_CONSTANT0_1  
#define LWE397_AT_CONSTANT0_1                     (0x609)
#define LWE397_AT_CONSTANT0_1_VAL                   19:0


// Register LWE397_AT_CONSTANT0_2  
#define LWE397_AT_CONSTANT0_2                     (0x60a)
#define LWE397_AT_CONSTANT0_2_VAL                   19:0


// Register LWE397_AT_CONSTANT0_3  
#define LWE397_AT_CONSTANT0_3                     (0x60b)
#define LWE397_AT_CONSTANT0_3_VAL                   19:0


// Register LWE397_AT_TRAM_OFFSET_0  
#define LWE397_AT_TRAM_OFFSET_0                   (0x60c)
#define LWE397_AT_TRAM_OFFSET_0_INDEX                       8:0


// Register LWE397_AT_TRAM_DATA_0  
#define LWE397_AT_TRAM_DATA_0                     (0x60d)
#define LWE397_AT_TRAM_DATA_0_VAL                   31:0


// Register LWE397_AT_CLKEN_OVERRIDE_0  
#define LWE397_AT_CLKEN_OVERRIDE_0                        (0x60e)
#define LWE397_AT_CLKEN_OVERRIDE_0_CORE_CLKEN_OVR                   0:0
#define LWE397_AT_CLKEN_OVERRIDE_0_CORE_CLKEN_OVR_CLK_GATED                       (0)
#define LWE397_AT_CLKEN_OVERRIDE_0_CORE_CLKEN_OVR_CLK_ALWAYS_ON                   (1)

#define LWE397_AT_CLKEN_OVERRIDE_0_SLOPES_CLKEN_OVR                 1:1
#define LWE397_AT_CLKEN_OVERRIDE_0_SLOPES_CLKEN_OVR_CLK_GATED                     (0)
#define LWE397_AT_CLKEN_OVERRIDE_0_SLOPES_CLKEN_OVR_CLK_ALWAYS_ON                 (1)

#define LWE397_AT_CLKEN_OVERRIDE_0_TRAM_CLKEN_OVR                   2:2
#define LWE397_AT_CLKEN_OVERRIDE_0_TRAM_CLKEN_OVR_CLK_GATED                       (0)
#define LWE397_AT_CLKEN_OVERRIDE_0_TRAM_CLKEN_OVR_CLK_ALWAYS_ON                   (1)

#define LWE397_AT_CLKEN_OVERRIDE_0_BARY_CLKEN_OVR                   3:3
#define LWE397_AT_CLKEN_OVERRIDE_0_BARY_CLKEN_OVR_CLK_GATED                       (0)
#define LWE397_AT_CLKEN_OVERRIDE_0_BARY_CLKEN_OVR_CLK_ALWAYS_ON                   (1)

#define LWE397_AT_CLKEN_OVERRIDE_0_ONEOVERW_CLKEN_OVR                       4:4
#define LWE397_AT_CLKEN_OVERRIDE_0_ONEOVERW_CLKEN_OVR_CLK_GATED                   (0)
#define LWE397_AT_CLKEN_OVERRIDE_0_ONEOVERW_CLKEN_OVR_CLK_ALWAYS_ON                       (1)

#define LWE397_AT_CLKEN_OVERRIDE_0_MFU_CLKEN_OVR                    5:5
#define LWE397_AT_CLKEN_OVERRIDE_0_MFU_CLKEN_OVR_CLK_GATED                        (0)
#define LWE397_AT_CLKEN_OVERRIDE_0_MFU_CLKEN_OVR_CLK_ALWAYS_ON                    (1)

#define LWE397_AT_CLKEN_OVERRIDE_0_MULABS_CLKEN_OVR                 6:6
#define LWE397_AT_CLKEN_OVERRIDE_0_MULABS_CLKEN_OVR_CLK_GATED                     (0)
#define LWE397_AT_CLKEN_OVERRIDE_0_MULABS_CLKEN_OVR_CLK_ALWAYS_ON                 (1)

#define LWE397_AT_CLKEN_OVERRIDE_0_IPAS_CLKEN_OVR                   7:7
#define LWE397_AT_CLKEN_OVERRIDE_0_IPAS_CLKEN_OVR_CLK_GATED                       (0)
#define LWE397_AT_CLKEN_OVERRIDE_0_IPAS_CLKEN_OVR_CLK_ALWAYS_ON                   (1)


// Register LWE397_TEX_INST_OFFSET_0  
#define LWE397_TEX_INST_OFFSET_0                  (0x700)
#define LWE397_TEX_INST_OFFSET_0_INDEX                      5:0


// Register LWE397_TEX_INST_DATA_0  
#define LWE397_TEX_INST_DATA_0                    (0x701)
#define LWE397_TEX_INST_DATA_0_USER_LOD                     13:13
#define LWE397_TEX_INST_DATA_0_USER_LOD_DISABLE                   (0)
#define LWE397_TEX_INST_DATA_0_USER_LOD_ENABLE                    (1)

#define LWE397_TEX_INST_DATA_0_USER_LOD_BIAS                        12:12
#define LWE397_TEX_INST_DATA_0_USER_LOD_BIAS_DISABLE                      (0)
#define LWE397_TEX_INST_DATA_0_USER_LOD_BIAS_ENABLE                       (1)

#define LWE397_TEX_INST_DATA_0_FETCH_KILLED                 11:11
#define LWE397_TEX_INST_DATA_0_FETCH_KILLED_DISABLE                       (0)
#define LWE397_TEX_INST_DATA_0_FETCH_KILLED_ENABLE                        (1)

#define LWE397_TEX_INST_DATA_0_FETCH                        10:10
#define LWE397_TEX_INST_DATA_0_FETCH_DISABLE                      (0)
#define LWE397_TEX_INST_DATA_0_FETCH_ENABLE                       (1)

#define LWE397_TEX_INST_DATA_0_COMP_SEL                     7:7
#define LWE397_TEX_INST_DATA_0_COMP_SEL_RG                        (0)
#define LWE397_TEX_INST_DATA_0_COMP_SEL_BA                        (1)

#define LWE397_TEX_INST_DATA_0_COLOR_KEY                    6:6
#define LWE397_TEX_INST_DATA_0_COLOR_KEY_DISABLE                  (0)
#define LWE397_TEX_INST_DATA_0_COLOR_KEY_ENABLE                   (1)

#define LWE397_TEX_INST_DATA_0_REG_DST                      5:5
#define LWE397_TEX_INST_DATA_0_REG_DST_R0R1                       (0)
#define LWE397_TEX_INST_DATA_0_REG_DST_R2R3                       (1)

#define LWE397_TEX_INST_DATA_0_REG_SRC                      4:4
#define LWE397_TEX_INST_DATA_0_REG_SRC_R0R1R2                     (0)
#define LWE397_TEX_INST_DATA_0_REG_SRC_R2R3R0                     (1)

#define LWE397_TEX_INST_DATA_0_SURF_PTR                     3:0


// Register LWE397_TEX_COLORKEY_0  
#define LWE397_TEX_COLORKEY_0                     (0x702)
#define LWE397_TEX_COLORKEY_0_VALUE                 31:0


// Register LWE397_TEX_TEXADDR_0  
#define LWE397_TEX_TEXADDR_0                      (0x710)
#define LWE397_TEX_TEXADDR_0_BASE_ADDRESS                   31:0


// Register LWE397_TEX_TEXADDR  
#define LWE397_TEX_TEXADDR                        (0x710)
#define LWE397_TEX_TEXADDR_BASE_ADDRESS                     31:0


// Register LWE397_TEX_TEXADDR_1  
#define LWE397_TEX_TEXADDR_1                      (0x711)
#define LWE397_TEX_TEXADDR_1_BASE_ADDRESS                   31:0


// Register LWE397_TEX_TEXADDR_2  
#define LWE397_TEX_TEXADDR_2                      (0x712)
#define LWE397_TEX_TEXADDR_2_BASE_ADDRESS                   31:0


// Register LWE397_TEX_TEXADDR_3  
#define LWE397_TEX_TEXADDR_3                      (0x713)
#define LWE397_TEX_TEXADDR_3_BASE_ADDRESS                   31:0


// Register LWE397_TEX_TEXADDR_4  
#define LWE397_TEX_TEXADDR_4                      (0x714)
#define LWE397_TEX_TEXADDR_4_BASE_ADDRESS                   31:0


// Register LWE397_TEX_TEXADDR_5  
#define LWE397_TEX_TEXADDR_5                      (0x715)
#define LWE397_TEX_TEXADDR_5_BASE_ADDRESS                   31:0


// Register LWE397_TEX_TEXADDR_6  
#define LWE397_TEX_TEXADDR_6                      (0x716)
#define LWE397_TEX_TEXADDR_6_BASE_ADDRESS                   31:0


// Register LWE397_TEX_TEXADDR_7  
#define LWE397_TEX_TEXADDR_7                      (0x717)
#define LWE397_TEX_TEXADDR_7_BASE_ADDRESS                   31:0


// Register LWE397_TEX_TEXADDR_8  
#define LWE397_TEX_TEXADDR_8                      (0x718)
#define LWE397_TEX_TEXADDR_8_BASE_ADDRESS                   31:0


// Register LWE397_TEX_TEXADDR_9  
#define LWE397_TEX_TEXADDR_9                      (0x719)
#define LWE397_TEX_TEXADDR_9_BASE_ADDRESS                   31:0


// Register LWE397_TEX_TEXADDR_10  
#define LWE397_TEX_TEXADDR_10                     (0x71a)
#define LWE397_TEX_TEXADDR_10_BASE_ADDRESS                  31:0


// Register LWE397_TEX_TEXADDR_11  
#define LWE397_TEX_TEXADDR_11                     (0x71b)
#define LWE397_TEX_TEXADDR_11_BASE_ADDRESS                  31:0


// Register LWE397_TEX_TEXADDR_12  
#define LWE397_TEX_TEXADDR_12                     (0x71c)
#define LWE397_TEX_TEXADDR_12_BASE_ADDRESS                  31:0


// Register LWE397_TEX_TEXADDR_13  
#define LWE397_TEX_TEXADDR_13                     (0x71d)
#define LWE397_TEX_TEXADDR_13_BASE_ADDRESS                  31:0


// Register LWE397_TEX_TEXADDR_14  
#define LWE397_TEX_TEXADDR_14                     (0x71e)
#define LWE397_TEX_TEXADDR_14_BASE_ADDRESS                  31:0


// Register LWE397_TEX_TEXADDR_15  
#define LWE397_TEX_TEXADDR_15                     (0x71f)
#define LWE397_TEX_TEXADDR_15_BASE_ADDRESS                  31:0


// Register LWE397_TEX_TEXDESC_0  
#define LWE397_TEX_TEXDESC_0                      (0x720)
#define LWE397_TEX_TEXDESC_0_WIDTH                  31:20

#define LWE397_TEX_TEXDESC_0_HEIGHT                 19:8

#define LWE397_TEX_TEXDESC_0_NORMALIZE                      7:7
#define LWE397_TEX_TEXDESC_0_NORMALIZE_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_0_NORMALIZE_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_0_LOG2_WIDTH                     31:28

#define LWE397_TEX_TEXDESC_0_LOG2_HEIGHT                    27:24

#define LWE397_TEX_TEXDESC_0_LOD_MIN                        23:16

#define LWE397_TEX_TEXDESC_0_LOD_MAX                        15:8

#define LWE397_TEX_TEXDESC_0_BASE_LEVEL_ONLY                        7:7
#define LWE397_TEX_TEXDESC_0_BASE_LEVEL_ONLY_DISABLE                      (0)
#define LWE397_TEX_TEXDESC_0_BASE_LEVEL_ONLY_ENABLE                       (1)

#define LWE397_TEX_TEXDESC_0_NON_POWER_OF_TWO                       6:6
#define LWE397_TEX_TEXDESC_0_NON_POWER_OF_TWO_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_0_NON_POWER_OF_TWO_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_0_ARRAY_MAX                      5:0

#define LWE397_TEX_TEXDESC_0_TRILINEAR_OPT                  30:30
#define LWE397_TEX_TEXDESC_0_TRILINEAR_OPT_DISABLE                        (0)
#define LWE397_TEX_TEXDESC_0_TRILINEAR_OPT_ENABLE                 (1)

#define LWE397_TEX_TEXDESC_0_LERP_MAG                       29:29

#define LWE397_TEX_TEXDESC_0_LERP_MIN                       28:28

#define LWE397_TEX_TEXDESC_0_LERP_MIP                       27:27

#define LWE397_TEX_TEXDESC_0_LOD_BIAS                       26:18

#define LWE397_TEX_TEXDESC_0_MAX_ANISO                      17:14

#define LWE397_TEX_TEXDESC_0_SURF_FORMAT                    13:8

#define LWE397_TEX_TEXDESC_0_LWBEMAP                        7:7

#define LWE397_TEX_TEXDESC_0_LAYOUT                 6:4
#define LWE397_TEX_TEXDESC_0_LAYOUT_LINEAR                        (0)
#define LWE397_TEX_TEXDESC_0_LAYOUT_SWIZZLED                      (1)
#define LWE397_TEX_TEXDESC_0_LAYOUT_TILED_LINEAR                  (2)
#define LWE397_TEX_TEXDESC_0_LAYOUT_TILED_SWIZZLED                        (3)
#define LWE397_TEX_TEXDESC_0_LAYOUT_XY_TILED_LINEAR                       (4)
#define LWE397_TEX_TEXDESC_0_LAYOUT_XY_TILED_SWIZZLED                     (5)

#define LWE397_TEX_TEXDESC_0_MIRROR_S                       3:3
#define LWE397_TEX_TEXDESC_0_MIRROR_S_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_0_MIRROR_S_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_0_MIRROR_T                       2:2
#define LWE397_TEX_TEXDESC_0_MIRROR_T_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_0_MIRROR_T_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_0_CLAMP_S                        1:1
#define LWE397_TEX_TEXDESC_0_CLAMP_S_WRAP                 (0)
#define LWE397_TEX_TEXDESC_0_CLAMP_S_CLAMP                        (1)

#define LWE397_TEX_TEXDESC_0_CLAMP_T                        0:0
#define LWE397_TEX_TEXDESC_0_CLAMP_T_WRAP                 (0)
#define LWE397_TEX_TEXDESC_0_CLAMP_T_CLAMP                        (1)


// Register LWE397_TEX_TEXDESC  
#define LWE397_TEX_TEXDESC                        (0x720)
#define LWE397_TEX_TEXDESC_WIDTH                    31:20

#define LWE397_TEX_TEXDESC_HEIGHT                   19:8

#define LWE397_TEX_TEXDESC_NORMALIZE                        7:7
#define LWE397_TEX_TEXDESC_NORMALIZE_DISABLE                      (0)
#define LWE397_TEX_TEXDESC_NORMALIZE_ENABLE                       (1)

#define LWE397_TEX_TEXDESC_LOG2_WIDTH                       31:28

#define LWE397_TEX_TEXDESC_LOG2_HEIGHT                      27:24

#define LWE397_TEX_TEXDESC_LOD_MIN                  23:16

#define LWE397_TEX_TEXDESC_LOD_MAX                  15:8

#define LWE397_TEX_TEXDESC_BASE_LEVEL_ONLY                  7:7
#define LWE397_TEX_TEXDESC_BASE_LEVEL_ONLY_DISABLE                        (0)
#define LWE397_TEX_TEXDESC_BASE_LEVEL_ONLY_ENABLE                 (1)

#define LWE397_TEX_TEXDESC_NON_POWER_OF_TWO                 6:6
#define LWE397_TEX_TEXDESC_NON_POWER_OF_TWO_DISABLE                       (0)
#define LWE397_TEX_TEXDESC_NON_POWER_OF_TWO_ENABLE                        (1)

#define LWE397_TEX_TEXDESC_ARRAY_MAX                        5:0

#define LWE397_TEX_TEXDESC_TRILINEAR_OPT                    30:30
#define LWE397_TEX_TEXDESC_TRILINEAR_OPT_DISABLE                  (0)
#define LWE397_TEX_TEXDESC_TRILINEAR_OPT_ENABLE                   (1)

#define LWE397_TEX_TEXDESC_LERP_MAG                 29:29

#define LWE397_TEX_TEXDESC_LERP_MIN                 28:28

#define LWE397_TEX_TEXDESC_LERP_MIP                 27:27

#define LWE397_TEX_TEXDESC_LOD_BIAS                 26:18

#define LWE397_TEX_TEXDESC_MAX_ANISO                        17:14

#define LWE397_TEX_TEXDESC_SURF_FORMAT                      13:8

#define LWE397_TEX_TEXDESC_LWBEMAP                  7:7

#define LWE397_TEX_TEXDESC_LAYOUT                   6:4
#define LWE397_TEX_TEXDESC_LAYOUT_LINEAR                  (0)
#define LWE397_TEX_TEXDESC_LAYOUT_SWIZZLED                        (1)
#define LWE397_TEX_TEXDESC_LAYOUT_TILED_LINEAR                    (2)
#define LWE397_TEX_TEXDESC_LAYOUT_TILED_SWIZZLED                  (3)
#define LWE397_TEX_TEXDESC_LAYOUT_XY_TILED_LINEAR                 (4)
#define LWE397_TEX_TEXDESC_LAYOUT_XY_TILED_SWIZZLED                       (5)

#define LWE397_TEX_TEXDESC_MIRROR_S                 3:3
#define LWE397_TEX_TEXDESC_MIRROR_S_DISABLE                       (0)
#define LWE397_TEX_TEXDESC_MIRROR_S_ENABLE                        (1)

#define LWE397_TEX_TEXDESC_MIRROR_T                 2:2
#define LWE397_TEX_TEXDESC_MIRROR_T_DISABLE                       (0)
#define LWE397_TEX_TEXDESC_MIRROR_T_ENABLE                        (1)

#define LWE397_TEX_TEXDESC_CLAMP_S                  1:1
#define LWE397_TEX_TEXDESC_CLAMP_S_WRAP                   (0)
#define LWE397_TEX_TEXDESC_CLAMP_S_CLAMP                  (1)

#define LWE397_TEX_TEXDESC_CLAMP_T                  0:0
#define LWE397_TEX_TEXDESC_CLAMP_T_WRAP                   (0)
#define LWE397_TEX_TEXDESC_CLAMP_T_CLAMP                  (1)


// Register LWE397_TEX_TEXDESC_1  
#define LWE397_TEX_TEXDESC_1                      (0x721)
#define LWE397_TEX_TEXDESC_1_WIDTH                  31:20

#define LWE397_TEX_TEXDESC_1_HEIGHT                 19:8

#define LWE397_TEX_TEXDESC_1_NORMALIZE                      7:7
#define LWE397_TEX_TEXDESC_1_NORMALIZE_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_1_NORMALIZE_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_1_LOG2_WIDTH                     31:28

#define LWE397_TEX_TEXDESC_1_LOG2_HEIGHT                    27:24

#define LWE397_TEX_TEXDESC_1_LOD_MIN                        23:16

#define LWE397_TEX_TEXDESC_1_LOD_MAX                        15:8

#define LWE397_TEX_TEXDESC_1_BASE_LEVEL_ONLY                        7:7
#define LWE397_TEX_TEXDESC_1_BASE_LEVEL_ONLY_DISABLE                      (0)
#define LWE397_TEX_TEXDESC_1_BASE_LEVEL_ONLY_ENABLE                       (1)

#define LWE397_TEX_TEXDESC_1_NON_POWER_OF_TWO                       6:6
#define LWE397_TEX_TEXDESC_1_NON_POWER_OF_TWO_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_1_NON_POWER_OF_TWO_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_1_ARRAY_MAX                      5:0

#define LWE397_TEX_TEXDESC_1_TRILINEAR_OPT                  30:30
#define LWE397_TEX_TEXDESC_1_TRILINEAR_OPT_DISABLE                        (0)
#define LWE397_TEX_TEXDESC_1_TRILINEAR_OPT_ENABLE                 (1)

#define LWE397_TEX_TEXDESC_1_LERP_MAG                       29:29

#define LWE397_TEX_TEXDESC_1_LERP_MIN                       28:28

#define LWE397_TEX_TEXDESC_1_LERP_MIP                       27:27

#define LWE397_TEX_TEXDESC_1_LOD_BIAS                       26:18

#define LWE397_TEX_TEXDESC_1_MAX_ANISO                      17:14

#define LWE397_TEX_TEXDESC_1_SURF_FORMAT                    13:8

#define LWE397_TEX_TEXDESC_1_LWBEMAP                        7:7

#define LWE397_TEX_TEXDESC_1_LAYOUT                 6:4
#define LWE397_TEX_TEXDESC_1_LAYOUT_LINEAR                        (0)
#define LWE397_TEX_TEXDESC_1_LAYOUT_SWIZZLED                      (1)
#define LWE397_TEX_TEXDESC_1_LAYOUT_TILED_LINEAR                  (2)
#define LWE397_TEX_TEXDESC_1_LAYOUT_TILED_SWIZZLED                        (3)
#define LWE397_TEX_TEXDESC_1_LAYOUT_XY_TILED_LINEAR                       (4)
#define LWE397_TEX_TEXDESC_1_LAYOUT_XY_TILED_SWIZZLED                     (5)

#define LWE397_TEX_TEXDESC_1_MIRROR_S                       3:3
#define LWE397_TEX_TEXDESC_1_MIRROR_S_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_1_MIRROR_S_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_1_MIRROR_T                       2:2
#define LWE397_TEX_TEXDESC_1_MIRROR_T_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_1_MIRROR_T_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_1_CLAMP_S                        1:1
#define LWE397_TEX_TEXDESC_1_CLAMP_S_WRAP                 (0)
#define LWE397_TEX_TEXDESC_1_CLAMP_S_CLAMP                        (1)

#define LWE397_TEX_TEXDESC_1_CLAMP_T                        0:0
#define LWE397_TEX_TEXDESC_1_CLAMP_T_WRAP                 (0)
#define LWE397_TEX_TEXDESC_1_CLAMP_T_CLAMP                        (1)


// Register LWE397_TEX_TEXDESC_2  
#define LWE397_TEX_TEXDESC_2                      (0x722)
#define LWE397_TEX_TEXDESC_2_WIDTH                  31:20

#define LWE397_TEX_TEXDESC_2_HEIGHT                 19:8

#define LWE397_TEX_TEXDESC_2_NORMALIZE                      7:7
#define LWE397_TEX_TEXDESC_2_NORMALIZE_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_2_NORMALIZE_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_2_LOG2_WIDTH                     31:28

#define LWE397_TEX_TEXDESC_2_LOG2_HEIGHT                    27:24

#define LWE397_TEX_TEXDESC_2_LOD_MIN                        23:16

#define LWE397_TEX_TEXDESC_2_LOD_MAX                        15:8

#define LWE397_TEX_TEXDESC_2_BASE_LEVEL_ONLY                        7:7
#define LWE397_TEX_TEXDESC_2_BASE_LEVEL_ONLY_DISABLE                      (0)
#define LWE397_TEX_TEXDESC_2_BASE_LEVEL_ONLY_ENABLE                       (1)

#define LWE397_TEX_TEXDESC_2_NON_POWER_OF_TWO                       6:6
#define LWE397_TEX_TEXDESC_2_NON_POWER_OF_TWO_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_2_NON_POWER_OF_TWO_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_2_ARRAY_MAX                      5:0

#define LWE397_TEX_TEXDESC_2_TRILINEAR_OPT                  30:30
#define LWE397_TEX_TEXDESC_2_TRILINEAR_OPT_DISABLE                        (0)
#define LWE397_TEX_TEXDESC_2_TRILINEAR_OPT_ENABLE                 (1)

#define LWE397_TEX_TEXDESC_2_LERP_MAG                       29:29

#define LWE397_TEX_TEXDESC_2_LERP_MIN                       28:28

#define LWE397_TEX_TEXDESC_2_LERP_MIP                       27:27

#define LWE397_TEX_TEXDESC_2_LOD_BIAS                       26:18

#define LWE397_TEX_TEXDESC_2_MAX_ANISO                      17:14

#define LWE397_TEX_TEXDESC_2_SURF_FORMAT                    13:8

#define LWE397_TEX_TEXDESC_2_LWBEMAP                        7:7

#define LWE397_TEX_TEXDESC_2_LAYOUT                 6:4
#define LWE397_TEX_TEXDESC_2_LAYOUT_LINEAR                        (0)
#define LWE397_TEX_TEXDESC_2_LAYOUT_SWIZZLED                      (1)
#define LWE397_TEX_TEXDESC_2_LAYOUT_TILED_LINEAR                  (2)
#define LWE397_TEX_TEXDESC_2_LAYOUT_TILED_SWIZZLED                        (3)
#define LWE397_TEX_TEXDESC_2_LAYOUT_XY_TILED_LINEAR                       (4)
#define LWE397_TEX_TEXDESC_2_LAYOUT_XY_TILED_SWIZZLED                     (5)

#define LWE397_TEX_TEXDESC_2_MIRROR_S                       3:3
#define LWE397_TEX_TEXDESC_2_MIRROR_S_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_2_MIRROR_S_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_2_MIRROR_T                       2:2
#define LWE397_TEX_TEXDESC_2_MIRROR_T_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_2_MIRROR_T_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_2_CLAMP_S                        1:1
#define LWE397_TEX_TEXDESC_2_CLAMP_S_WRAP                 (0)
#define LWE397_TEX_TEXDESC_2_CLAMP_S_CLAMP                        (1)

#define LWE397_TEX_TEXDESC_2_CLAMP_T                        0:0
#define LWE397_TEX_TEXDESC_2_CLAMP_T_WRAP                 (0)
#define LWE397_TEX_TEXDESC_2_CLAMP_T_CLAMP                        (1)


// Register LWE397_TEX_TEXDESC_3  
#define LWE397_TEX_TEXDESC_3                      (0x723)
#define LWE397_TEX_TEXDESC_3_WIDTH                  31:20

#define LWE397_TEX_TEXDESC_3_HEIGHT                 19:8

#define LWE397_TEX_TEXDESC_3_NORMALIZE                      7:7
#define LWE397_TEX_TEXDESC_3_NORMALIZE_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_3_NORMALIZE_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_3_LOG2_WIDTH                     31:28

#define LWE397_TEX_TEXDESC_3_LOG2_HEIGHT                    27:24

#define LWE397_TEX_TEXDESC_3_LOD_MIN                        23:16

#define LWE397_TEX_TEXDESC_3_LOD_MAX                        15:8

#define LWE397_TEX_TEXDESC_3_BASE_LEVEL_ONLY                        7:7
#define LWE397_TEX_TEXDESC_3_BASE_LEVEL_ONLY_DISABLE                      (0)
#define LWE397_TEX_TEXDESC_3_BASE_LEVEL_ONLY_ENABLE                       (1)

#define LWE397_TEX_TEXDESC_3_NON_POWER_OF_TWO                       6:6
#define LWE397_TEX_TEXDESC_3_NON_POWER_OF_TWO_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_3_NON_POWER_OF_TWO_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_3_ARRAY_MAX                      5:0

#define LWE397_TEX_TEXDESC_3_TRILINEAR_OPT                  30:30
#define LWE397_TEX_TEXDESC_3_TRILINEAR_OPT_DISABLE                        (0)
#define LWE397_TEX_TEXDESC_3_TRILINEAR_OPT_ENABLE                 (1)

#define LWE397_TEX_TEXDESC_3_LERP_MAG                       29:29

#define LWE397_TEX_TEXDESC_3_LERP_MIN                       28:28

#define LWE397_TEX_TEXDESC_3_LERP_MIP                       27:27

#define LWE397_TEX_TEXDESC_3_LOD_BIAS                       26:18

#define LWE397_TEX_TEXDESC_3_MAX_ANISO                      17:14

#define LWE397_TEX_TEXDESC_3_SURF_FORMAT                    13:8

#define LWE397_TEX_TEXDESC_3_LWBEMAP                        7:7

#define LWE397_TEX_TEXDESC_3_LAYOUT                 6:4
#define LWE397_TEX_TEXDESC_3_LAYOUT_LINEAR                        (0)
#define LWE397_TEX_TEXDESC_3_LAYOUT_SWIZZLED                      (1)
#define LWE397_TEX_TEXDESC_3_LAYOUT_TILED_LINEAR                  (2)
#define LWE397_TEX_TEXDESC_3_LAYOUT_TILED_SWIZZLED                        (3)
#define LWE397_TEX_TEXDESC_3_LAYOUT_XY_TILED_LINEAR                       (4)
#define LWE397_TEX_TEXDESC_3_LAYOUT_XY_TILED_SWIZZLED                     (5)

#define LWE397_TEX_TEXDESC_3_MIRROR_S                       3:3
#define LWE397_TEX_TEXDESC_3_MIRROR_S_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_3_MIRROR_S_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_3_MIRROR_T                       2:2
#define LWE397_TEX_TEXDESC_3_MIRROR_T_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_3_MIRROR_T_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_3_CLAMP_S                        1:1
#define LWE397_TEX_TEXDESC_3_CLAMP_S_WRAP                 (0)
#define LWE397_TEX_TEXDESC_3_CLAMP_S_CLAMP                        (1)

#define LWE397_TEX_TEXDESC_3_CLAMP_T                        0:0
#define LWE397_TEX_TEXDESC_3_CLAMP_T_WRAP                 (0)
#define LWE397_TEX_TEXDESC_3_CLAMP_T_CLAMP                        (1)


// Register LWE397_TEX_TEXDESC_4  
#define LWE397_TEX_TEXDESC_4                      (0x724)
#define LWE397_TEX_TEXDESC_4_WIDTH                  31:20

#define LWE397_TEX_TEXDESC_4_HEIGHT                 19:8

#define LWE397_TEX_TEXDESC_4_NORMALIZE                      7:7
#define LWE397_TEX_TEXDESC_4_NORMALIZE_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_4_NORMALIZE_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_4_LOG2_WIDTH                     31:28

#define LWE397_TEX_TEXDESC_4_LOG2_HEIGHT                    27:24

#define LWE397_TEX_TEXDESC_4_LOD_MIN                        23:16

#define LWE397_TEX_TEXDESC_4_LOD_MAX                        15:8

#define LWE397_TEX_TEXDESC_4_BASE_LEVEL_ONLY                        7:7
#define LWE397_TEX_TEXDESC_4_BASE_LEVEL_ONLY_DISABLE                      (0)
#define LWE397_TEX_TEXDESC_4_BASE_LEVEL_ONLY_ENABLE                       (1)

#define LWE397_TEX_TEXDESC_4_NON_POWER_OF_TWO                       6:6
#define LWE397_TEX_TEXDESC_4_NON_POWER_OF_TWO_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_4_NON_POWER_OF_TWO_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_4_ARRAY_MAX                      5:0

#define LWE397_TEX_TEXDESC_4_TRILINEAR_OPT                  30:30
#define LWE397_TEX_TEXDESC_4_TRILINEAR_OPT_DISABLE                        (0)
#define LWE397_TEX_TEXDESC_4_TRILINEAR_OPT_ENABLE                 (1)

#define LWE397_TEX_TEXDESC_4_LERP_MAG                       29:29

#define LWE397_TEX_TEXDESC_4_LERP_MIN                       28:28

#define LWE397_TEX_TEXDESC_4_LERP_MIP                       27:27

#define LWE397_TEX_TEXDESC_4_LOD_BIAS                       26:18

#define LWE397_TEX_TEXDESC_4_MAX_ANISO                      17:14

#define LWE397_TEX_TEXDESC_4_SURF_FORMAT                    13:8

#define LWE397_TEX_TEXDESC_4_LWBEMAP                        7:7

#define LWE397_TEX_TEXDESC_4_LAYOUT                 6:4
#define LWE397_TEX_TEXDESC_4_LAYOUT_LINEAR                        (0)
#define LWE397_TEX_TEXDESC_4_LAYOUT_SWIZZLED                      (1)
#define LWE397_TEX_TEXDESC_4_LAYOUT_TILED_LINEAR                  (2)
#define LWE397_TEX_TEXDESC_4_LAYOUT_TILED_SWIZZLED                        (3)
#define LWE397_TEX_TEXDESC_4_LAYOUT_XY_TILED_LINEAR                       (4)
#define LWE397_TEX_TEXDESC_4_LAYOUT_XY_TILED_SWIZZLED                     (5)

#define LWE397_TEX_TEXDESC_4_MIRROR_S                       3:3
#define LWE397_TEX_TEXDESC_4_MIRROR_S_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_4_MIRROR_S_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_4_MIRROR_T                       2:2
#define LWE397_TEX_TEXDESC_4_MIRROR_T_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_4_MIRROR_T_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_4_CLAMP_S                        1:1
#define LWE397_TEX_TEXDESC_4_CLAMP_S_WRAP                 (0)
#define LWE397_TEX_TEXDESC_4_CLAMP_S_CLAMP                        (1)

#define LWE397_TEX_TEXDESC_4_CLAMP_T                        0:0
#define LWE397_TEX_TEXDESC_4_CLAMP_T_WRAP                 (0)
#define LWE397_TEX_TEXDESC_4_CLAMP_T_CLAMP                        (1)


// Register LWE397_TEX_TEXDESC_5  
#define LWE397_TEX_TEXDESC_5                      (0x725)
#define LWE397_TEX_TEXDESC_5_WIDTH                  31:20

#define LWE397_TEX_TEXDESC_5_HEIGHT                 19:8

#define LWE397_TEX_TEXDESC_5_NORMALIZE                      7:7
#define LWE397_TEX_TEXDESC_5_NORMALIZE_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_5_NORMALIZE_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_5_LOG2_WIDTH                     31:28

#define LWE397_TEX_TEXDESC_5_LOG2_HEIGHT                    27:24

#define LWE397_TEX_TEXDESC_5_LOD_MIN                        23:16

#define LWE397_TEX_TEXDESC_5_LOD_MAX                        15:8

#define LWE397_TEX_TEXDESC_5_BASE_LEVEL_ONLY                        7:7
#define LWE397_TEX_TEXDESC_5_BASE_LEVEL_ONLY_DISABLE                      (0)
#define LWE397_TEX_TEXDESC_5_BASE_LEVEL_ONLY_ENABLE                       (1)

#define LWE397_TEX_TEXDESC_5_NON_POWER_OF_TWO                       6:6
#define LWE397_TEX_TEXDESC_5_NON_POWER_OF_TWO_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_5_NON_POWER_OF_TWO_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_5_ARRAY_MAX                      5:0

#define LWE397_TEX_TEXDESC_5_TRILINEAR_OPT                  30:30
#define LWE397_TEX_TEXDESC_5_TRILINEAR_OPT_DISABLE                        (0)
#define LWE397_TEX_TEXDESC_5_TRILINEAR_OPT_ENABLE                 (1)

#define LWE397_TEX_TEXDESC_5_LERP_MAG                       29:29

#define LWE397_TEX_TEXDESC_5_LERP_MIN                       28:28

#define LWE397_TEX_TEXDESC_5_LERP_MIP                       27:27

#define LWE397_TEX_TEXDESC_5_LOD_BIAS                       26:18

#define LWE397_TEX_TEXDESC_5_MAX_ANISO                      17:14

#define LWE397_TEX_TEXDESC_5_SURF_FORMAT                    13:8

#define LWE397_TEX_TEXDESC_5_LWBEMAP                        7:7

#define LWE397_TEX_TEXDESC_5_LAYOUT                 6:4
#define LWE397_TEX_TEXDESC_5_LAYOUT_LINEAR                        (0)
#define LWE397_TEX_TEXDESC_5_LAYOUT_SWIZZLED                      (1)
#define LWE397_TEX_TEXDESC_5_LAYOUT_TILED_LINEAR                  (2)
#define LWE397_TEX_TEXDESC_5_LAYOUT_TILED_SWIZZLED                        (3)
#define LWE397_TEX_TEXDESC_5_LAYOUT_XY_TILED_LINEAR                       (4)
#define LWE397_TEX_TEXDESC_5_LAYOUT_XY_TILED_SWIZZLED                     (5)

#define LWE397_TEX_TEXDESC_5_MIRROR_S                       3:3
#define LWE397_TEX_TEXDESC_5_MIRROR_S_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_5_MIRROR_S_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_5_MIRROR_T                       2:2
#define LWE397_TEX_TEXDESC_5_MIRROR_T_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_5_MIRROR_T_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_5_CLAMP_S                        1:1
#define LWE397_TEX_TEXDESC_5_CLAMP_S_WRAP                 (0)
#define LWE397_TEX_TEXDESC_5_CLAMP_S_CLAMP                        (1)

#define LWE397_TEX_TEXDESC_5_CLAMP_T                        0:0
#define LWE397_TEX_TEXDESC_5_CLAMP_T_WRAP                 (0)
#define LWE397_TEX_TEXDESC_5_CLAMP_T_CLAMP                        (1)


// Register LWE397_TEX_TEXDESC_6  
#define LWE397_TEX_TEXDESC_6                      (0x726)
#define LWE397_TEX_TEXDESC_6_WIDTH                  31:20

#define LWE397_TEX_TEXDESC_6_HEIGHT                 19:8

#define LWE397_TEX_TEXDESC_6_NORMALIZE                      7:7
#define LWE397_TEX_TEXDESC_6_NORMALIZE_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_6_NORMALIZE_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_6_LOG2_WIDTH                     31:28

#define LWE397_TEX_TEXDESC_6_LOG2_HEIGHT                    27:24

#define LWE397_TEX_TEXDESC_6_LOD_MIN                        23:16

#define LWE397_TEX_TEXDESC_6_LOD_MAX                        15:8

#define LWE397_TEX_TEXDESC_6_BASE_LEVEL_ONLY                        7:7
#define LWE397_TEX_TEXDESC_6_BASE_LEVEL_ONLY_DISABLE                      (0)
#define LWE397_TEX_TEXDESC_6_BASE_LEVEL_ONLY_ENABLE                       (1)

#define LWE397_TEX_TEXDESC_6_NON_POWER_OF_TWO                       6:6
#define LWE397_TEX_TEXDESC_6_NON_POWER_OF_TWO_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_6_NON_POWER_OF_TWO_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_6_ARRAY_MAX                      5:0

#define LWE397_TEX_TEXDESC_6_TRILINEAR_OPT                  30:30
#define LWE397_TEX_TEXDESC_6_TRILINEAR_OPT_DISABLE                        (0)
#define LWE397_TEX_TEXDESC_6_TRILINEAR_OPT_ENABLE                 (1)

#define LWE397_TEX_TEXDESC_6_LERP_MAG                       29:29

#define LWE397_TEX_TEXDESC_6_LERP_MIN                       28:28

#define LWE397_TEX_TEXDESC_6_LERP_MIP                       27:27

#define LWE397_TEX_TEXDESC_6_LOD_BIAS                       26:18

#define LWE397_TEX_TEXDESC_6_MAX_ANISO                      17:14

#define LWE397_TEX_TEXDESC_6_SURF_FORMAT                    13:8

#define LWE397_TEX_TEXDESC_6_LWBEMAP                        7:7

#define LWE397_TEX_TEXDESC_6_LAYOUT                 6:4
#define LWE397_TEX_TEXDESC_6_LAYOUT_LINEAR                        (0)
#define LWE397_TEX_TEXDESC_6_LAYOUT_SWIZZLED                      (1)
#define LWE397_TEX_TEXDESC_6_LAYOUT_TILED_LINEAR                  (2)
#define LWE397_TEX_TEXDESC_6_LAYOUT_TILED_SWIZZLED                        (3)
#define LWE397_TEX_TEXDESC_6_LAYOUT_XY_TILED_LINEAR                       (4)
#define LWE397_TEX_TEXDESC_6_LAYOUT_XY_TILED_SWIZZLED                     (5)

#define LWE397_TEX_TEXDESC_6_MIRROR_S                       3:3
#define LWE397_TEX_TEXDESC_6_MIRROR_S_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_6_MIRROR_S_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_6_MIRROR_T                       2:2
#define LWE397_TEX_TEXDESC_6_MIRROR_T_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_6_MIRROR_T_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_6_CLAMP_S                        1:1
#define LWE397_TEX_TEXDESC_6_CLAMP_S_WRAP                 (0)
#define LWE397_TEX_TEXDESC_6_CLAMP_S_CLAMP                        (1)

#define LWE397_TEX_TEXDESC_6_CLAMP_T                        0:0
#define LWE397_TEX_TEXDESC_6_CLAMP_T_WRAP                 (0)
#define LWE397_TEX_TEXDESC_6_CLAMP_T_CLAMP                        (1)


// Register LWE397_TEX_TEXDESC_7  
#define LWE397_TEX_TEXDESC_7                      (0x727)
#define LWE397_TEX_TEXDESC_7_WIDTH                  31:20

#define LWE397_TEX_TEXDESC_7_HEIGHT                 19:8

#define LWE397_TEX_TEXDESC_7_NORMALIZE                      7:7
#define LWE397_TEX_TEXDESC_7_NORMALIZE_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_7_NORMALIZE_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_7_LOG2_WIDTH                     31:28

#define LWE397_TEX_TEXDESC_7_LOG2_HEIGHT                    27:24

#define LWE397_TEX_TEXDESC_7_LOD_MIN                        23:16

#define LWE397_TEX_TEXDESC_7_LOD_MAX                        15:8

#define LWE397_TEX_TEXDESC_7_BASE_LEVEL_ONLY                        7:7
#define LWE397_TEX_TEXDESC_7_BASE_LEVEL_ONLY_DISABLE                      (0)
#define LWE397_TEX_TEXDESC_7_BASE_LEVEL_ONLY_ENABLE                       (1)

#define LWE397_TEX_TEXDESC_7_NON_POWER_OF_TWO                       6:6
#define LWE397_TEX_TEXDESC_7_NON_POWER_OF_TWO_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_7_NON_POWER_OF_TWO_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_7_ARRAY_MAX                      5:0

#define LWE397_TEX_TEXDESC_7_TRILINEAR_OPT                  30:30
#define LWE397_TEX_TEXDESC_7_TRILINEAR_OPT_DISABLE                        (0)
#define LWE397_TEX_TEXDESC_7_TRILINEAR_OPT_ENABLE                 (1)

#define LWE397_TEX_TEXDESC_7_LERP_MAG                       29:29

#define LWE397_TEX_TEXDESC_7_LERP_MIN                       28:28

#define LWE397_TEX_TEXDESC_7_LERP_MIP                       27:27

#define LWE397_TEX_TEXDESC_7_LOD_BIAS                       26:18

#define LWE397_TEX_TEXDESC_7_MAX_ANISO                      17:14

#define LWE397_TEX_TEXDESC_7_SURF_FORMAT                    13:8

#define LWE397_TEX_TEXDESC_7_LWBEMAP                        7:7

#define LWE397_TEX_TEXDESC_7_LAYOUT                 6:4
#define LWE397_TEX_TEXDESC_7_LAYOUT_LINEAR                        (0)
#define LWE397_TEX_TEXDESC_7_LAYOUT_SWIZZLED                      (1)
#define LWE397_TEX_TEXDESC_7_LAYOUT_TILED_LINEAR                  (2)
#define LWE397_TEX_TEXDESC_7_LAYOUT_TILED_SWIZZLED                        (3)
#define LWE397_TEX_TEXDESC_7_LAYOUT_XY_TILED_LINEAR                       (4)
#define LWE397_TEX_TEXDESC_7_LAYOUT_XY_TILED_SWIZZLED                     (5)

#define LWE397_TEX_TEXDESC_7_MIRROR_S                       3:3
#define LWE397_TEX_TEXDESC_7_MIRROR_S_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_7_MIRROR_S_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_7_MIRROR_T                       2:2
#define LWE397_TEX_TEXDESC_7_MIRROR_T_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_7_MIRROR_T_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_7_CLAMP_S                        1:1
#define LWE397_TEX_TEXDESC_7_CLAMP_S_WRAP                 (0)
#define LWE397_TEX_TEXDESC_7_CLAMP_S_CLAMP                        (1)

#define LWE397_TEX_TEXDESC_7_CLAMP_T                        0:0
#define LWE397_TEX_TEXDESC_7_CLAMP_T_WRAP                 (0)
#define LWE397_TEX_TEXDESC_7_CLAMP_T_CLAMP                        (1)


// Register LWE397_TEX_TEXDESC_8  
#define LWE397_TEX_TEXDESC_8                      (0x728)
#define LWE397_TEX_TEXDESC_8_WIDTH                  31:20

#define LWE397_TEX_TEXDESC_8_HEIGHT                 19:8

#define LWE397_TEX_TEXDESC_8_NORMALIZE                      7:7
#define LWE397_TEX_TEXDESC_8_NORMALIZE_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_8_NORMALIZE_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_8_LOG2_WIDTH                     31:28

#define LWE397_TEX_TEXDESC_8_LOG2_HEIGHT                    27:24

#define LWE397_TEX_TEXDESC_8_LOD_MIN                        23:16

#define LWE397_TEX_TEXDESC_8_LOD_MAX                        15:8

#define LWE397_TEX_TEXDESC_8_BASE_LEVEL_ONLY                        7:7
#define LWE397_TEX_TEXDESC_8_BASE_LEVEL_ONLY_DISABLE                      (0)
#define LWE397_TEX_TEXDESC_8_BASE_LEVEL_ONLY_ENABLE                       (1)

#define LWE397_TEX_TEXDESC_8_NON_POWER_OF_TWO                       6:6
#define LWE397_TEX_TEXDESC_8_NON_POWER_OF_TWO_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_8_NON_POWER_OF_TWO_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_8_ARRAY_MAX                      5:0

#define LWE397_TEX_TEXDESC_8_TRILINEAR_OPT                  30:30
#define LWE397_TEX_TEXDESC_8_TRILINEAR_OPT_DISABLE                        (0)
#define LWE397_TEX_TEXDESC_8_TRILINEAR_OPT_ENABLE                 (1)

#define LWE397_TEX_TEXDESC_8_LERP_MAG                       29:29

#define LWE397_TEX_TEXDESC_8_LERP_MIN                       28:28

#define LWE397_TEX_TEXDESC_8_LERP_MIP                       27:27

#define LWE397_TEX_TEXDESC_8_LOD_BIAS                       26:18

#define LWE397_TEX_TEXDESC_8_MAX_ANISO                      17:14

#define LWE397_TEX_TEXDESC_8_SURF_FORMAT                    13:8

#define LWE397_TEX_TEXDESC_8_LWBEMAP                        7:7

#define LWE397_TEX_TEXDESC_8_LAYOUT                 6:4
#define LWE397_TEX_TEXDESC_8_LAYOUT_LINEAR                        (0)
#define LWE397_TEX_TEXDESC_8_LAYOUT_SWIZZLED                      (1)
#define LWE397_TEX_TEXDESC_8_LAYOUT_TILED_LINEAR                  (2)
#define LWE397_TEX_TEXDESC_8_LAYOUT_TILED_SWIZZLED                        (3)
#define LWE397_TEX_TEXDESC_8_LAYOUT_XY_TILED_LINEAR                       (4)
#define LWE397_TEX_TEXDESC_8_LAYOUT_XY_TILED_SWIZZLED                     (5)

#define LWE397_TEX_TEXDESC_8_MIRROR_S                       3:3
#define LWE397_TEX_TEXDESC_8_MIRROR_S_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_8_MIRROR_S_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_8_MIRROR_T                       2:2
#define LWE397_TEX_TEXDESC_8_MIRROR_T_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_8_MIRROR_T_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_8_CLAMP_S                        1:1
#define LWE397_TEX_TEXDESC_8_CLAMP_S_WRAP                 (0)
#define LWE397_TEX_TEXDESC_8_CLAMP_S_CLAMP                        (1)

#define LWE397_TEX_TEXDESC_8_CLAMP_T                        0:0
#define LWE397_TEX_TEXDESC_8_CLAMP_T_WRAP                 (0)
#define LWE397_TEX_TEXDESC_8_CLAMP_T_CLAMP                        (1)


// Register LWE397_TEX_TEXDESC_9  
#define LWE397_TEX_TEXDESC_9                      (0x729)
#define LWE397_TEX_TEXDESC_9_WIDTH                  31:20

#define LWE397_TEX_TEXDESC_9_HEIGHT                 19:8

#define LWE397_TEX_TEXDESC_9_NORMALIZE                      7:7
#define LWE397_TEX_TEXDESC_9_NORMALIZE_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_9_NORMALIZE_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_9_LOG2_WIDTH                     31:28

#define LWE397_TEX_TEXDESC_9_LOG2_HEIGHT                    27:24

#define LWE397_TEX_TEXDESC_9_LOD_MIN                        23:16

#define LWE397_TEX_TEXDESC_9_LOD_MAX                        15:8

#define LWE397_TEX_TEXDESC_9_BASE_LEVEL_ONLY                        7:7
#define LWE397_TEX_TEXDESC_9_BASE_LEVEL_ONLY_DISABLE                      (0)
#define LWE397_TEX_TEXDESC_9_BASE_LEVEL_ONLY_ENABLE                       (1)

#define LWE397_TEX_TEXDESC_9_NON_POWER_OF_TWO                       6:6
#define LWE397_TEX_TEXDESC_9_NON_POWER_OF_TWO_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_9_NON_POWER_OF_TWO_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_9_ARRAY_MAX                      5:0

#define LWE397_TEX_TEXDESC_9_TRILINEAR_OPT                  30:30
#define LWE397_TEX_TEXDESC_9_TRILINEAR_OPT_DISABLE                        (0)
#define LWE397_TEX_TEXDESC_9_TRILINEAR_OPT_ENABLE                 (1)

#define LWE397_TEX_TEXDESC_9_LERP_MAG                       29:29

#define LWE397_TEX_TEXDESC_9_LERP_MIN                       28:28

#define LWE397_TEX_TEXDESC_9_LERP_MIP                       27:27

#define LWE397_TEX_TEXDESC_9_LOD_BIAS                       26:18

#define LWE397_TEX_TEXDESC_9_MAX_ANISO                      17:14

#define LWE397_TEX_TEXDESC_9_SURF_FORMAT                    13:8

#define LWE397_TEX_TEXDESC_9_LWBEMAP                        7:7

#define LWE397_TEX_TEXDESC_9_LAYOUT                 6:4
#define LWE397_TEX_TEXDESC_9_LAYOUT_LINEAR                        (0)
#define LWE397_TEX_TEXDESC_9_LAYOUT_SWIZZLED                      (1)
#define LWE397_TEX_TEXDESC_9_LAYOUT_TILED_LINEAR                  (2)
#define LWE397_TEX_TEXDESC_9_LAYOUT_TILED_SWIZZLED                        (3)
#define LWE397_TEX_TEXDESC_9_LAYOUT_XY_TILED_LINEAR                       (4)
#define LWE397_TEX_TEXDESC_9_LAYOUT_XY_TILED_SWIZZLED                     (5)

#define LWE397_TEX_TEXDESC_9_MIRROR_S                       3:3
#define LWE397_TEX_TEXDESC_9_MIRROR_S_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_9_MIRROR_S_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_9_MIRROR_T                       2:2
#define LWE397_TEX_TEXDESC_9_MIRROR_T_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_9_MIRROR_T_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_9_CLAMP_S                        1:1
#define LWE397_TEX_TEXDESC_9_CLAMP_S_WRAP                 (0)
#define LWE397_TEX_TEXDESC_9_CLAMP_S_CLAMP                        (1)

#define LWE397_TEX_TEXDESC_9_CLAMP_T                        0:0
#define LWE397_TEX_TEXDESC_9_CLAMP_T_WRAP                 (0)
#define LWE397_TEX_TEXDESC_9_CLAMP_T_CLAMP                        (1)


// Register LWE397_TEX_TEXDESC_10  
#define LWE397_TEX_TEXDESC_10                     (0x72a)
#define LWE397_TEX_TEXDESC_10_WIDTH                 31:20

#define LWE397_TEX_TEXDESC_10_HEIGHT                        19:8

#define LWE397_TEX_TEXDESC_10_NORMALIZE                     7:7
#define LWE397_TEX_TEXDESC_10_NORMALIZE_DISABLE                   (0)
#define LWE397_TEX_TEXDESC_10_NORMALIZE_ENABLE                    (1)

#define LWE397_TEX_TEXDESC_10_LOG2_WIDTH                    31:28

#define LWE397_TEX_TEXDESC_10_LOG2_HEIGHT                   27:24

#define LWE397_TEX_TEXDESC_10_LOD_MIN                       23:16

#define LWE397_TEX_TEXDESC_10_LOD_MAX                       15:8

#define LWE397_TEX_TEXDESC_10_BASE_LEVEL_ONLY                       7:7
#define LWE397_TEX_TEXDESC_10_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_10_BASE_LEVEL_ONLY_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_10_NON_POWER_OF_TWO                      6:6
#define LWE397_TEX_TEXDESC_10_NON_POWER_OF_TWO_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_10_NON_POWER_OF_TWO_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_10_ARRAY_MAX                     5:0

#define LWE397_TEX_TEXDESC_10_TRILINEAR_OPT                 30:30
#define LWE397_TEX_TEXDESC_10_TRILINEAR_OPT_DISABLE                       (0)
#define LWE397_TEX_TEXDESC_10_TRILINEAR_OPT_ENABLE                        (1)

#define LWE397_TEX_TEXDESC_10_LERP_MAG                      29:29

#define LWE397_TEX_TEXDESC_10_LERP_MIN                      28:28

#define LWE397_TEX_TEXDESC_10_LERP_MIP                      27:27

#define LWE397_TEX_TEXDESC_10_LOD_BIAS                      26:18

#define LWE397_TEX_TEXDESC_10_MAX_ANISO                     17:14

#define LWE397_TEX_TEXDESC_10_SURF_FORMAT                   13:8

#define LWE397_TEX_TEXDESC_10_LWBEMAP                       7:7

#define LWE397_TEX_TEXDESC_10_LAYOUT                        6:4
#define LWE397_TEX_TEXDESC_10_LAYOUT_LINEAR                       (0)
#define LWE397_TEX_TEXDESC_10_LAYOUT_SWIZZLED                     (1)
#define LWE397_TEX_TEXDESC_10_LAYOUT_TILED_LINEAR                 (2)
#define LWE397_TEX_TEXDESC_10_LAYOUT_TILED_SWIZZLED                       (3)
#define LWE397_TEX_TEXDESC_10_LAYOUT_XY_TILED_LINEAR                      (4)
#define LWE397_TEX_TEXDESC_10_LAYOUT_XY_TILED_SWIZZLED                    (5)

#define LWE397_TEX_TEXDESC_10_MIRROR_S                      3:3
#define LWE397_TEX_TEXDESC_10_MIRROR_S_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_10_MIRROR_S_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_10_MIRROR_T                      2:2
#define LWE397_TEX_TEXDESC_10_MIRROR_T_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_10_MIRROR_T_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_10_CLAMP_S                       1:1
#define LWE397_TEX_TEXDESC_10_CLAMP_S_WRAP                        (0)
#define LWE397_TEX_TEXDESC_10_CLAMP_S_CLAMP                       (1)

#define LWE397_TEX_TEXDESC_10_CLAMP_T                       0:0
#define LWE397_TEX_TEXDESC_10_CLAMP_T_WRAP                        (0)
#define LWE397_TEX_TEXDESC_10_CLAMP_T_CLAMP                       (1)


// Register LWE397_TEX_TEXDESC_11  
#define LWE397_TEX_TEXDESC_11                     (0x72b)
#define LWE397_TEX_TEXDESC_11_WIDTH                 31:20

#define LWE397_TEX_TEXDESC_11_HEIGHT                        19:8

#define LWE397_TEX_TEXDESC_11_NORMALIZE                     7:7
#define LWE397_TEX_TEXDESC_11_NORMALIZE_DISABLE                   (0)
#define LWE397_TEX_TEXDESC_11_NORMALIZE_ENABLE                    (1)

#define LWE397_TEX_TEXDESC_11_LOG2_WIDTH                    31:28

#define LWE397_TEX_TEXDESC_11_LOG2_HEIGHT                   27:24

#define LWE397_TEX_TEXDESC_11_LOD_MIN                       23:16

#define LWE397_TEX_TEXDESC_11_LOD_MAX                       15:8

#define LWE397_TEX_TEXDESC_11_BASE_LEVEL_ONLY                       7:7
#define LWE397_TEX_TEXDESC_11_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_11_BASE_LEVEL_ONLY_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_11_NON_POWER_OF_TWO                      6:6
#define LWE397_TEX_TEXDESC_11_NON_POWER_OF_TWO_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_11_NON_POWER_OF_TWO_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_11_ARRAY_MAX                     5:0

#define LWE397_TEX_TEXDESC_11_TRILINEAR_OPT                 30:30
#define LWE397_TEX_TEXDESC_11_TRILINEAR_OPT_DISABLE                       (0)
#define LWE397_TEX_TEXDESC_11_TRILINEAR_OPT_ENABLE                        (1)

#define LWE397_TEX_TEXDESC_11_LERP_MAG                      29:29

#define LWE397_TEX_TEXDESC_11_LERP_MIN                      28:28

#define LWE397_TEX_TEXDESC_11_LERP_MIP                      27:27

#define LWE397_TEX_TEXDESC_11_LOD_BIAS                      26:18

#define LWE397_TEX_TEXDESC_11_MAX_ANISO                     17:14

#define LWE397_TEX_TEXDESC_11_SURF_FORMAT                   13:8

#define LWE397_TEX_TEXDESC_11_LWBEMAP                       7:7

#define LWE397_TEX_TEXDESC_11_LAYOUT                        6:4
#define LWE397_TEX_TEXDESC_11_LAYOUT_LINEAR                       (0)
#define LWE397_TEX_TEXDESC_11_LAYOUT_SWIZZLED                     (1)
#define LWE397_TEX_TEXDESC_11_LAYOUT_TILED_LINEAR                 (2)
#define LWE397_TEX_TEXDESC_11_LAYOUT_TILED_SWIZZLED                       (3)
#define LWE397_TEX_TEXDESC_11_LAYOUT_XY_TILED_LINEAR                      (4)
#define LWE397_TEX_TEXDESC_11_LAYOUT_XY_TILED_SWIZZLED                    (5)

#define LWE397_TEX_TEXDESC_11_MIRROR_S                      3:3
#define LWE397_TEX_TEXDESC_11_MIRROR_S_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_11_MIRROR_S_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_11_MIRROR_T                      2:2
#define LWE397_TEX_TEXDESC_11_MIRROR_T_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_11_MIRROR_T_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_11_CLAMP_S                       1:1
#define LWE397_TEX_TEXDESC_11_CLAMP_S_WRAP                        (0)
#define LWE397_TEX_TEXDESC_11_CLAMP_S_CLAMP                       (1)

#define LWE397_TEX_TEXDESC_11_CLAMP_T                       0:0
#define LWE397_TEX_TEXDESC_11_CLAMP_T_WRAP                        (0)
#define LWE397_TEX_TEXDESC_11_CLAMP_T_CLAMP                       (1)


// Register LWE397_TEX_TEXDESC_12  
#define LWE397_TEX_TEXDESC_12                     (0x72c)
#define LWE397_TEX_TEXDESC_12_WIDTH                 31:20

#define LWE397_TEX_TEXDESC_12_HEIGHT                        19:8

#define LWE397_TEX_TEXDESC_12_NORMALIZE                     7:7
#define LWE397_TEX_TEXDESC_12_NORMALIZE_DISABLE                   (0)
#define LWE397_TEX_TEXDESC_12_NORMALIZE_ENABLE                    (1)

#define LWE397_TEX_TEXDESC_12_LOG2_WIDTH                    31:28

#define LWE397_TEX_TEXDESC_12_LOG2_HEIGHT                   27:24

#define LWE397_TEX_TEXDESC_12_LOD_MIN                       23:16

#define LWE397_TEX_TEXDESC_12_LOD_MAX                       15:8

#define LWE397_TEX_TEXDESC_12_BASE_LEVEL_ONLY                       7:7
#define LWE397_TEX_TEXDESC_12_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_12_BASE_LEVEL_ONLY_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_12_NON_POWER_OF_TWO                      6:6
#define LWE397_TEX_TEXDESC_12_NON_POWER_OF_TWO_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_12_NON_POWER_OF_TWO_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_12_ARRAY_MAX                     5:0

#define LWE397_TEX_TEXDESC_12_TRILINEAR_OPT                 30:30
#define LWE397_TEX_TEXDESC_12_TRILINEAR_OPT_DISABLE                       (0)
#define LWE397_TEX_TEXDESC_12_TRILINEAR_OPT_ENABLE                        (1)

#define LWE397_TEX_TEXDESC_12_LERP_MAG                      29:29

#define LWE397_TEX_TEXDESC_12_LERP_MIN                      28:28

#define LWE397_TEX_TEXDESC_12_LERP_MIP                      27:27

#define LWE397_TEX_TEXDESC_12_LOD_BIAS                      26:18

#define LWE397_TEX_TEXDESC_12_MAX_ANISO                     17:14

#define LWE397_TEX_TEXDESC_12_SURF_FORMAT                   13:8

#define LWE397_TEX_TEXDESC_12_LWBEMAP                       7:7

#define LWE397_TEX_TEXDESC_12_LAYOUT                        6:4
#define LWE397_TEX_TEXDESC_12_LAYOUT_LINEAR                       (0)
#define LWE397_TEX_TEXDESC_12_LAYOUT_SWIZZLED                     (1)
#define LWE397_TEX_TEXDESC_12_LAYOUT_TILED_LINEAR                 (2)
#define LWE397_TEX_TEXDESC_12_LAYOUT_TILED_SWIZZLED                       (3)
#define LWE397_TEX_TEXDESC_12_LAYOUT_XY_TILED_LINEAR                      (4)
#define LWE397_TEX_TEXDESC_12_LAYOUT_XY_TILED_SWIZZLED                    (5)

#define LWE397_TEX_TEXDESC_12_MIRROR_S                      3:3
#define LWE397_TEX_TEXDESC_12_MIRROR_S_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_12_MIRROR_S_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_12_MIRROR_T                      2:2
#define LWE397_TEX_TEXDESC_12_MIRROR_T_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_12_MIRROR_T_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_12_CLAMP_S                       1:1
#define LWE397_TEX_TEXDESC_12_CLAMP_S_WRAP                        (0)
#define LWE397_TEX_TEXDESC_12_CLAMP_S_CLAMP                       (1)

#define LWE397_TEX_TEXDESC_12_CLAMP_T                       0:0
#define LWE397_TEX_TEXDESC_12_CLAMP_T_WRAP                        (0)
#define LWE397_TEX_TEXDESC_12_CLAMP_T_CLAMP                       (1)


// Register LWE397_TEX_TEXDESC_13  
#define LWE397_TEX_TEXDESC_13                     (0x72d)
#define LWE397_TEX_TEXDESC_13_WIDTH                 31:20

#define LWE397_TEX_TEXDESC_13_HEIGHT                        19:8

#define LWE397_TEX_TEXDESC_13_NORMALIZE                     7:7
#define LWE397_TEX_TEXDESC_13_NORMALIZE_DISABLE                   (0)
#define LWE397_TEX_TEXDESC_13_NORMALIZE_ENABLE                    (1)

#define LWE397_TEX_TEXDESC_13_LOG2_WIDTH                    31:28

#define LWE397_TEX_TEXDESC_13_LOG2_HEIGHT                   27:24

#define LWE397_TEX_TEXDESC_13_LOD_MIN                       23:16

#define LWE397_TEX_TEXDESC_13_LOD_MAX                       15:8

#define LWE397_TEX_TEXDESC_13_BASE_LEVEL_ONLY                       7:7
#define LWE397_TEX_TEXDESC_13_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_13_BASE_LEVEL_ONLY_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_13_NON_POWER_OF_TWO                      6:6
#define LWE397_TEX_TEXDESC_13_NON_POWER_OF_TWO_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_13_NON_POWER_OF_TWO_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_13_ARRAY_MAX                     5:0

#define LWE397_TEX_TEXDESC_13_TRILINEAR_OPT                 30:30
#define LWE397_TEX_TEXDESC_13_TRILINEAR_OPT_DISABLE                       (0)
#define LWE397_TEX_TEXDESC_13_TRILINEAR_OPT_ENABLE                        (1)

#define LWE397_TEX_TEXDESC_13_LERP_MAG                      29:29

#define LWE397_TEX_TEXDESC_13_LERP_MIN                      28:28

#define LWE397_TEX_TEXDESC_13_LERP_MIP                      27:27

#define LWE397_TEX_TEXDESC_13_LOD_BIAS                      26:18

#define LWE397_TEX_TEXDESC_13_MAX_ANISO                     17:14

#define LWE397_TEX_TEXDESC_13_SURF_FORMAT                   13:8

#define LWE397_TEX_TEXDESC_13_LWBEMAP                       7:7

#define LWE397_TEX_TEXDESC_13_LAYOUT                        6:4
#define LWE397_TEX_TEXDESC_13_LAYOUT_LINEAR                       (0)
#define LWE397_TEX_TEXDESC_13_LAYOUT_SWIZZLED                     (1)
#define LWE397_TEX_TEXDESC_13_LAYOUT_TILED_LINEAR                 (2)
#define LWE397_TEX_TEXDESC_13_LAYOUT_TILED_SWIZZLED                       (3)
#define LWE397_TEX_TEXDESC_13_LAYOUT_XY_TILED_LINEAR                      (4)
#define LWE397_TEX_TEXDESC_13_LAYOUT_XY_TILED_SWIZZLED                    (5)

#define LWE397_TEX_TEXDESC_13_MIRROR_S                      3:3
#define LWE397_TEX_TEXDESC_13_MIRROR_S_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_13_MIRROR_S_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_13_MIRROR_T                      2:2
#define LWE397_TEX_TEXDESC_13_MIRROR_T_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_13_MIRROR_T_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_13_CLAMP_S                       1:1
#define LWE397_TEX_TEXDESC_13_CLAMP_S_WRAP                        (0)
#define LWE397_TEX_TEXDESC_13_CLAMP_S_CLAMP                       (1)

#define LWE397_TEX_TEXDESC_13_CLAMP_T                       0:0
#define LWE397_TEX_TEXDESC_13_CLAMP_T_WRAP                        (0)
#define LWE397_TEX_TEXDESC_13_CLAMP_T_CLAMP                       (1)


// Register LWE397_TEX_TEXDESC_14  
#define LWE397_TEX_TEXDESC_14                     (0x72e)
#define LWE397_TEX_TEXDESC_14_WIDTH                 31:20

#define LWE397_TEX_TEXDESC_14_HEIGHT                        19:8

#define LWE397_TEX_TEXDESC_14_NORMALIZE                     7:7
#define LWE397_TEX_TEXDESC_14_NORMALIZE_DISABLE                   (0)
#define LWE397_TEX_TEXDESC_14_NORMALIZE_ENABLE                    (1)

#define LWE397_TEX_TEXDESC_14_LOG2_WIDTH                    31:28

#define LWE397_TEX_TEXDESC_14_LOG2_HEIGHT                   27:24

#define LWE397_TEX_TEXDESC_14_LOD_MIN                       23:16

#define LWE397_TEX_TEXDESC_14_LOD_MAX                       15:8

#define LWE397_TEX_TEXDESC_14_BASE_LEVEL_ONLY                       7:7
#define LWE397_TEX_TEXDESC_14_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_14_BASE_LEVEL_ONLY_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_14_NON_POWER_OF_TWO                      6:6
#define LWE397_TEX_TEXDESC_14_NON_POWER_OF_TWO_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_14_NON_POWER_OF_TWO_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_14_ARRAY_MAX                     5:0

#define LWE397_TEX_TEXDESC_14_TRILINEAR_OPT                 30:30
#define LWE397_TEX_TEXDESC_14_TRILINEAR_OPT_DISABLE                       (0)
#define LWE397_TEX_TEXDESC_14_TRILINEAR_OPT_ENABLE                        (1)

#define LWE397_TEX_TEXDESC_14_LERP_MAG                      29:29

#define LWE397_TEX_TEXDESC_14_LERP_MIN                      28:28

#define LWE397_TEX_TEXDESC_14_LERP_MIP                      27:27

#define LWE397_TEX_TEXDESC_14_LOD_BIAS                      26:18

#define LWE397_TEX_TEXDESC_14_MAX_ANISO                     17:14

#define LWE397_TEX_TEXDESC_14_SURF_FORMAT                   13:8

#define LWE397_TEX_TEXDESC_14_LWBEMAP                       7:7

#define LWE397_TEX_TEXDESC_14_LAYOUT                        6:4
#define LWE397_TEX_TEXDESC_14_LAYOUT_LINEAR                       (0)
#define LWE397_TEX_TEXDESC_14_LAYOUT_SWIZZLED                     (1)
#define LWE397_TEX_TEXDESC_14_LAYOUT_TILED_LINEAR                 (2)
#define LWE397_TEX_TEXDESC_14_LAYOUT_TILED_SWIZZLED                       (3)
#define LWE397_TEX_TEXDESC_14_LAYOUT_XY_TILED_LINEAR                      (4)
#define LWE397_TEX_TEXDESC_14_LAYOUT_XY_TILED_SWIZZLED                    (5)

#define LWE397_TEX_TEXDESC_14_MIRROR_S                      3:3
#define LWE397_TEX_TEXDESC_14_MIRROR_S_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_14_MIRROR_S_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_14_MIRROR_T                      2:2
#define LWE397_TEX_TEXDESC_14_MIRROR_T_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_14_MIRROR_T_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_14_CLAMP_S                       1:1
#define LWE397_TEX_TEXDESC_14_CLAMP_S_WRAP                        (0)
#define LWE397_TEX_TEXDESC_14_CLAMP_S_CLAMP                       (1)

#define LWE397_TEX_TEXDESC_14_CLAMP_T                       0:0
#define LWE397_TEX_TEXDESC_14_CLAMP_T_WRAP                        (0)
#define LWE397_TEX_TEXDESC_14_CLAMP_T_CLAMP                       (1)


// Register LWE397_TEX_TEXDESC_15  
#define LWE397_TEX_TEXDESC_15                     (0x72f)
#define LWE397_TEX_TEXDESC_15_WIDTH                 31:20

#define LWE397_TEX_TEXDESC_15_HEIGHT                        19:8

#define LWE397_TEX_TEXDESC_15_NORMALIZE                     7:7
#define LWE397_TEX_TEXDESC_15_NORMALIZE_DISABLE                   (0)
#define LWE397_TEX_TEXDESC_15_NORMALIZE_ENABLE                    (1)

#define LWE397_TEX_TEXDESC_15_LOG2_WIDTH                    31:28

#define LWE397_TEX_TEXDESC_15_LOG2_HEIGHT                   27:24

#define LWE397_TEX_TEXDESC_15_LOD_MIN                       23:16

#define LWE397_TEX_TEXDESC_15_LOD_MAX                       15:8

#define LWE397_TEX_TEXDESC_15_BASE_LEVEL_ONLY                       7:7
#define LWE397_TEX_TEXDESC_15_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_15_BASE_LEVEL_ONLY_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_15_NON_POWER_OF_TWO                      6:6
#define LWE397_TEX_TEXDESC_15_NON_POWER_OF_TWO_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_15_NON_POWER_OF_TWO_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_15_ARRAY_MAX                     5:0

#define LWE397_TEX_TEXDESC_15_TRILINEAR_OPT                 30:30
#define LWE397_TEX_TEXDESC_15_TRILINEAR_OPT_DISABLE                       (0)
#define LWE397_TEX_TEXDESC_15_TRILINEAR_OPT_ENABLE                        (1)

#define LWE397_TEX_TEXDESC_15_LERP_MAG                      29:29

#define LWE397_TEX_TEXDESC_15_LERP_MIN                      28:28

#define LWE397_TEX_TEXDESC_15_LERP_MIP                      27:27

#define LWE397_TEX_TEXDESC_15_LOD_BIAS                      26:18

#define LWE397_TEX_TEXDESC_15_MAX_ANISO                     17:14

#define LWE397_TEX_TEXDESC_15_SURF_FORMAT                   13:8

#define LWE397_TEX_TEXDESC_15_LWBEMAP                       7:7

#define LWE397_TEX_TEXDESC_15_LAYOUT                        6:4
#define LWE397_TEX_TEXDESC_15_LAYOUT_LINEAR                       (0)
#define LWE397_TEX_TEXDESC_15_LAYOUT_SWIZZLED                     (1)
#define LWE397_TEX_TEXDESC_15_LAYOUT_TILED_LINEAR                 (2)
#define LWE397_TEX_TEXDESC_15_LAYOUT_TILED_SWIZZLED                       (3)
#define LWE397_TEX_TEXDESC_15_LAYOUT_XY_TILED_LINEAR                      (4)
#define LWE397_TEX_TEXDESC_15_LAYOUT_XY_TILED_SWIZZLED                    (5)

#define LWE397_TEX_TEXDESC_15_MIRROR_S                      3:3
#define LWE397_TEX_TEXDESC_15_MIRROR_S_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_15_MIRROR_S_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_15_MIRROR_T                      2:2
#define LWE397_TEX_TEXDESC_15_MIRROR_T_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_15_MIRROR_T_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_15_CLAMP_S                       1:1
#define LWE397_TEX_TEXDESC_15_CLAMP_S_WRAP                        (0)
#define LWE397_TEX_TEXDESC_15_CLAMP_S_CLAMP                       (1)

#define LWE397_TEX_TEXDESC_15_CLAMP_T                       0:0
#define LWE397_TEX_TEXDESC_15_CLAMP_T_WRAP                        (0)
#define LWE397_TEX_TEXDESC_15_CLAMP_T_CLAMP                       (1)


// Register LWE397_TEX_TEXDESC_16  
#define LWE397_TEX_TEXDESC_16                     (0x730)
#define LWE397_TEX_TEXDESC_16_WIDTH                 31:20

#define LWE397_TEX_TEXDESC_16_HEIGHT                        19:8

#define LWE397_TEX_TEXDESC_16_NORMALIZE                     7:7
#define LWE397_TEX_TEXDESC_16_NORMALIZE_DISABLE                   (0)
#define LWE397_TEX_TEXDESC_16_NORMALIZE_ENABLE                    (1)

#define LWE397_TEX_TEXDESC_16_LOG2_WIDTH                    31:28

#define LWE397_TEX_TEXDESC_16_LOG2_HEIGHT                   27:24

#define LWE397_TEX_TEXDESC_16_LOD_MIN                       23:16

#define LWE397_TEX_TEXDESC_16_LOD_MAX                       15:8

#define LWE397_TEX_TEXDESC_16_BASE_LEVEL_ONLY                       7:7
#define LWE397_TEX_TEXDESC_16_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_16_BASE_LEVEL_ONLY_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_16_NON_POWER_OF_TWO                      6:6
#define LWE397_TEX_TEXDESC_16_NON_POWER_OF_TWO_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_16_NON_POWER_OF_TWO_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_16_ARRAY_MAX                     5:0

#define LWE397_TEX_TEXDESC_16_TRILINEAR_OPT                 30:30
#define LWE397_TEX_TEXDESC_16_TRILINEAR_OPT_DISABLE                       (0)
#define LWE397_TEX_TEXDESC_16_TRILINEAR_OPT_ENABLE                        (1)

#define LWE397_TEX_TEXDESC_16_LERP_MAG                      29:29

#define LWE397_TEX_TEXDESC_16_LERP_MIN                      28:28

#define LWE397_TEX_TEXDESC_16_LERP_MIP                      27:27

#define LWE397_TEX_TEXDESC_16_LOD_BIAS                      26:18

#define LWE397_TEX_TEXDESC_16_MAX_ANISO                     17:14

#define LWE397_TEX_TEXDESC_16_SURF_FORMAT                   13:8

#define LWE397_TEX_TEXDESC_16_LWBEMAP                       7:7

#define LWE397_TEX_TEXDESC_16_LAYOUT                        6:4
#define LWE397_TEX_TEXDESC_16_LAYOUT_LINEAR                       (0)
#define LWE397_TEX_TEXDESC_16_LAYOUT_SWIZZLED                     (1)
#define LWE397_TEX_TEXDESC_16_LAYOUT_TILED_LINEAR                 (2)
#define LWE397_TEX_TEXDESC_16_LAYOUT_TILED_SWIZZLED                       (3)
#define LWE397_TEX_TEXDESC_16_LAYOUT_XY_TILED_LINEAR                      (4)
#define LWE397_TEX_TEXDESC_16_LAYOUT_XY_TILED_SWIZZLED                    (5)

#define LWE397_TEX_TEXDESC_16_MIRROR_S                      3:3
#define LWE397_TEX_TEXDESC_16_MIRROR_S_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_16_MIRROR_S_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_16_MIRROR_T                      2:2
#define LWE397_TEX_TEXDESC_16_MIRROR_T_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_16_MIRROR_T_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_16_CLAMP_S                       1:1
#define LWE397_TEX_TEXDESC_16_CLAMP_S_WRAP                        (0)
#define LWE397_TEX_TEXDESC_16_CLAMP_S_CLAMP                       (1)

#define LWE397_TEX_TEXDESC_16_CLAMP_T                       0:0
#define LWE397_TEX_TEXDESC_16_CLAMP_T_WRAP                        (0)
#define LWE397_TEX_TEXDESC_16_CLAMP_T_CLAMP                       (1)


// Register LWE397_TEX_TEXDESC_17  
#define LWE397_TEX_TEXDESC_17                     (0x731)
#define LWE397_TEX_TEXDESC_17_WIDTH                 31:20

#define LWE397_TEX_TEXDESC_17_HEIGHT                        19:8

#define LWE397_TEX_TEXDESC_17_NORMALIZE                     7:7
#define LWE397_TEX_TEXDESC_17_NORMALIZE_DISABLE                   (0)
#define LWE397_TEX_TEXDESC_17_NORMALIZE_ENABLE                    (1)

#define LWE397_TEX_TEXDESC_17_LOG2_WIDTH                    31:28

#define LWE397_TEX_TEXDESC_17_LOG2_HEIGHT                   27:24

#define LWE397_TEX_TEXDESC_17_LOD_MIN                       23:16

#define LWE397_TEX_TEXDESC_17_LOD_MAX                       15:8

#define LWE397_TEX_TEXDESC_17_BASE_LEVEL_ONLY                       7:7
#define LWE397_TEX_TEXDESC_17_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_17_BASE_LEVEL_ONLY_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_17_NON_POWER_OF_TWO                      6:6
#define LWE397_TEX_TEXDESC_17_NON_POWER_OF_TWO_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_17_NON_POWER_OF_TWO_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_17_ARRAY_MAX                     5:0

#define LWE397_TEX_TEXDESC_17_TRILINEAR_OPT                 30:30
#define LWE397_TEX_TEXDESC_17_TRILINEAR_OPT_DISABLE                       (0)
#define LWE397_TEX_TEXDESC_17_TRILINEAR_OPT_ENABLE                        (1)

#define LWE397_TEX_TEXDESC_17_LERP_MAG                      29:29

#define LWE397_TEX_TEXDESC_17_LERP_MIN                      28:28

#define LWE397_TEX_TEXDESC_17_LERP_MIP                      27:27

#define LWE397_TEX_TEXDESC_17_LOD_BIAS                      26:18

#define LWE397_TEX_TEXDESC_17_MAX_ANISO                     17:14

#define LWE397_TEX_TEXDESC_17_SURF_FORMAT                   13:8

#define LWE397_TEX_TEXDESC_17_LWBEMAP                       7:7

#define LWE397_TEX_TEXDESC_17_LAYOUT                        6:4
#define LWE397_TEX_TEXDESC_17_LAYOUT_LINEAR                       (0)
#define LWE397_TEX_TEXDESC_17_LAYOUT_SWIZZLED                     (1)
#define LWE397_TEX_TEXDESC_17_LAYOUT_TILED_LINEAR                 (2)
#define LWE397_TEX_TEXDESC_17_LAYOUT_TILED_SWIZZLED                       (3)
#define LWE397_TEX_TEXDESC_17_LAYOUT_XY_TILED_LINEAR                      (4)
#define LWE397_TEX_TEXDESC_17_LAYOUT_XY_TILED_SWIZZLED                    (5)

#define LWE397_TEX_TEXDESC_17_MIRROR_S                      3:3
#define LWE397_TEX_TEXDESC_17_MIRROR_S_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_17_MIRROR_S_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_17_MIRROR_T                      2:2
#define LWE397_TEX_TEXDESC_17_MIRROR_T_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_17_MIRROR_T_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_17_CLAMP_S                       1:1
#define LWE397_TEX_TEXDESC_17_CLAMP_S_WRAP                        (0)
#define LWE397_TEX_TEXDESC_17_CLAMP_S_CLAMP                       (1)

#define LWE397_TEX_TEXDESC_17_CLAMP_T                       0:0
#define LWE397_TEX_TEXDESC_17_CLAMP_T_WRAP                        (0)
#define LWE397_TEX_TEXDESC_17_CLAMP_T_CLAMP                       (1)


// Register LWE397_TEX_TEXDESC_18  
#define LWE397_TEX_TEXDESC_18                     (0x732)
#define LWE397_TEX_TEXDESC_18_WIDTH                 31:20

#define LWE397_TEX_TEXDESC_18_HEIGHT                        19:8

#define LWE397_TEX_TEXDESC_18_NORMALIZE                     7:7
#define LWE397_TEX_TEXDESC_18_NORMALIZE_DISABLE                   (0)
#define LWE397_TEX_TEXDESC_18_NORMALIZE_ENABLE                    (1)

#define LWE397_TEX_TEXDESC_18_LOG2_WIDTH                    31:28

#define LWE397_TEX_TEXDESC_18_LOG2_HEIGHT                   27:24

#define LWE397_TEX_TEXDESC_18_LOD_MIN                       23:16

#define LWE397_TEX_TEXDESC_18_LOD_MAX                       15:8

#define LWE397_TEX_TEXDESC_18_BASE_LEVEL_ONLY                       7:7
#define LWE397_TEX_TEXDESC_18_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_18_BASE_LEVEL_ONLY_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_18_NON_POWER_OF_TWO                      6:6
#define LWE397_TEX_TEXDESC_18_NON_POWER_OF_TWO_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_18_NON_POWER_OF_TWO_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_18_ARRAY_MAX                     5:0

#define LWE397_TEX_TEXDESC_18_TRILINEAR_OPT                 30:30
#define LWE397_TEX_TEXDESC_18_TRILINEAR_OPT_DISABLE                       (0)
#define LWE397_TEX_TEXDESC_18_TRILINEAR_OPT_ENABLE                        (1)

#define LWE397_TEX_TEXDESC_18_LERP_MAG                      29:29

#define LWE397_TEX_TEXDESC_18_LERP_MIN                      28:28

#define LWE397_TEX_TEXDESC_18_LERP_MIP                      27:27

#define LWE397_TEX_TEXDESC_18_LOD_BIAS                      26:18

#define LWE397_TEX_TEXDESC_18_MAX_ANISO                     17:14

#define LWE397_TEX_TEXDESC_18_SURF_FORMAT                   13:8

#define LWE397_TEX_TEXDESC_18_LWBEMAP                       7:7

#define LWE397_TEX_TEXDESC_18_LAYOUT                        6:4
#define LWE397_TEX_TEXDESC_18_LAYOUT_LINEAR                       (0)
#define LWE397_TEX_TEXDESC_18_LAYOUT_SWIZZLED                     (1)
#define LWE397_TEX_TEXDESC_18_LAYOUT_TILED_LINEAR                 (2)
#define LWE397_TEX_TEXDESC_18_LAYOUT_TILED_SWIZZLED                       (3)
#define LWE397_TEX_TEXDESC_18_LAYOUT_XY_TILED_LINEAR                      (4)
#define LWE397_TEX_TEXDESC_18_LAYOUT_XY_TILED_SWIZZLED                    (5)

#define LWE397_TEX_TEXDESC_18_MIRROR_S                      3:3
#define LWE397_TEX_TEXDESC_18_MIRROR_S_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_18_MIRROR_S_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_18_MIRROR_T                      2:2
#define LWE397_TEX_TEXDESC_18_MIRROR_T_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_18_MIRROR_T_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_18_CLAMP_S                       1:1
#define LWE397_TEX_TEXDESC_18_CLAMP_S_WRAP                        (0)
#define LWE397_TEX_TEXDESC_18_CLAMP_S_CLAMP                       (1)

#define LWE397_TEX_TEXDESC_18_CLAMP_T                       0:0
#define LWE397_TEX_TEXDESC_18_CLAMP_T_WRAP                        (0)
#define LWE397_TEX_TEXDESC_18_CLAMP_T_CLAMP                       (1)


// Register LWE397_TEX_TEXDESC_19  
#define LWE397_TEX_TEXDESC_19                     (0x733)
#define LWE397_TEX_TEXDESC_19_WIDTH                 31:20

#define LWE397_TEX_TEXDESC_19_HEIGHT                        19:8

#define LWE397_TEX_TEXDESC_19_NORMALIZE                     7:7
#define LWE397_TEX_TEXDESC_19_NORMALIZE_DISABLE                   (0)
#define LWE397_TEX_TEXDESC_19_NORMALIZE_ENABLE                    (1)

#define LWE397_TEX_TEXDESC_19_LOG2_WIDTH                    31:28

#define LWE397_TEX_TEXDESC_19_LOG2_HEIGHT                   27:24

#define LWE397_TEX_TEXDESC_19_LOD_MIN                       23:16

#define LWE397_TEX_TEXDESC_19_LOD_MAX                       15:8

#define LWE397_TEX_TEXDESC_19_BASE_LEVEL_ONLY                       7:7
#define LWE397_TEX_TEXDESC_19_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_19_BASE_LEVEL_ONLY_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_19_NON_POWER_OF_TWO                      6:6
#define LWE397_TEX_TEXDESC_19_NON_POWER_OF_TWO_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_19_NON_POWER_OF_TWO_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_19_ARRAY_MAX                     5:0

#define LWE397_TEX_TEXDESC_19_TRILINEAR_OPT                 30:30
#define LWE397_TEX_TEXDESC_19_TRILINEAR_OPT_DISABLE                       (0)
#define LWE397_TEX_TEXDESC_19_TRILINEAR_OPT_ENABLE                        (1)

#define LWE397_TEX_TEXDESC_19_LERP_MAG                      29:29

#define LWE397_TEX_TEXDESC_19_LERP_MIN                      28:28

#define LWE397_TEX_TEXDESC_19_LERP_MIP                      27:27

#define LWE397_TEX_TEXDESC_19_LOD_BIAS                      26:18

#define LWE397_TEX_TEXDESC_19_MAX_ANISO                     17:14

#define LWE397_TEX_TEXDESC_19_SURF_FORMAT                   13:8

#define LWE397_TEX_TEXDESC_19_LWBEMAP                       7:7

#define LWE397_TEX_TEXDESC_19_LAYOUT                        6:4
#define LWE397_TEX_TEXDESC_19_LAYOUT_LINEAR                       (0)
#define LWE397_TEX_TEXDESC_19_LAYOUT_SWIZZLED                     (1)
#define LWE397_TEX_TEXDESC_19_LAYOUT_TILED_LINEAR                 (2)
#define LWE397_TEX_TEXDESC_19_LAYOUT_TILED_SWIZZLED                       (3)
#define LWE397_TEX_TEXDESC_19_LAYOUT_XY_TILED_LINEAR                      (4)
#define LWE397_TEX_TEXDESC_19_LAYOUT_XY_TILED_SWIZZLED                    (5)

#define LWE397_TEX_TEXDESC_19_MIRROR_S                      3:3
#define LWE397_TEX_TEXDESC_19_MIRROR_S_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_19_MIRROR_S_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_19_MIRROR_T                      2:2
#define LWE397_TEX_TEXDESC_19_MIRROR_T_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_19_MIRROR_T_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_19_CLAMP_S                       1:1
#define LWE397_TEX_TEXDESC_19_CLAMP_S_WRAP                        (0)
#define LWE397_TEX_TEXDESC_19_CLAMP_S_CLAMP                       (1)

#define LWE397_TEX_TEXDESC_19_CLAMP_T                       0:0
#define LWE397_TEX_TEXDESC_19_CLAMP_T_WRAP                        (0)
#define LWE397_TEX_TEXDESC_19_CLAMP_T_CLAMP                       (1)


// Register LWE397_TEX_TEXDESC_20  
#define LWE397_TEX_TEXDESC_20                     (0x734)
#define LWE397_TEX_TEXDESC_20_WIDTH                 31:20

#define LWE397_TEX_TEXDESC_20_HEIGHT                        19:8

#define LWE397_TEX_TEXDESC_20_NORMALIZE                     7:7
#define LWE397_TEX_TEXDESC_20_NORMALIZE_DISABLE                   (0)
#define LWE397_TEX_TEXDESC_20_NORMALIZE_ENABLE                    (1)

#define LWE397_TEX_TEXDESC_20_LOG2_WIDTH                    31:28

#define LWE397_TEX_TEXDESC_20_LOG2_HEIGHT                   27:24

#define LWE397_TEX_TEXDESC_20_LOD_MIN                       23:16

#define LWE397_TEX_TEXDESC_20_LOD_MAX                       15:8

#define LWE397_TEX_TEXDESC_20_BASE_LEVEL_ONLY                       7:7
#define LWE397_TEX_TEXDESC_20_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_20_BASE_LEVEL_ONLY_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_20_NON_POWER_OF_TWO                      6:6
#define LWE397_TEX_TEXDESC_20_NON_POWER_OF_TWO_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_20_NON_POWER_OF_TWO_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_20_ARRAY_MAX                     5:0

#define LWE397_TEX_TEXDESC_20_TRILINEAR_OPT                 30:30
#define LWE397_TEX_TEXDESC_20_TRILINEAR_OPT_DISABLE                       (0)
#define LWE397_TEX_TEXDESC_20_TRILINEAR_OPT_ENABLE                        (1)

#define LWE397_TEX_TEXDESC_20_LERP_MAG                      29:29

#define LWE397_TEX_TEXDESC_20_LERP_MIN                      28:28

#define LWE397_TEX_TEXDESC_20_LERP_MIP                      27:27

#define LWE397_TEX_TEXDESC_20_LOD_BIAS                      26:18

#define LWE397_TEX_TEXDESC_20_MAX_ANISO                     17:14

#define LWE397_TEX_TEXDESC_20_SURF_FORMAT                   13:8

#define LWE397_TEX_TEXDESC_20_LWBEMAP                       7:7

#define LWE397_TEX_TEXDESC_20_LAYOUT                        6:4
#define LWE397_TEX_TEXDESC_20_LAYOUT_LINEAR                       (0)
#define LWE397_TEX_TEXDESC_20_LAYOUT_SWIZZLED                     (1)
#define LWE397_TEX_TEXDESC_20_LAYOUT_TILED_LINEAR                 (2)
#define LWE397_TEX_TEXDESC_20_LAYOUT_TILED_SWIZZLED                       (3)
#define LWE397_TEX_TEXDESC_20_LAYOUT_XY_TILED_LINEAR                      (4)
#define LWE397_TEX_TEXDESC_20_LAYOUT_XY_TILED_SWIZZLED                    (5)

#define LWE397_TEX_TEXDESC_20_MIRROR_S                      3:3
#define LWE397_TEX_TEXDESC_20_MIRROR_S_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_20_MIRROR_S_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_20_MIRROR_T                      2:2
#define LWE397_TEX_TEXDESC_20_MIRROR_T_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_20_MIRROR_T_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_20_CLAMP_S                       1:1
#define LWE397_TEX_TEXDESC_20_CLAMP_S_WRAP                        (0)
#define LWE397_TEX_TEXDESC_20_CLAMP_S_CLAMP                       (1)

#define LWE397_TEX_TEXDESC_20_CLAMP_T                       0:0
#define LWE397_TEX_TEXDESC_20_CLAMP_T_WRAP                        (0)
#define LWE397_TEX_TEXDESC_20_CLAMP_T_CLAMP                       (1)


// Register LWE397_TEX_TEXDESC_21  
#define LWE397_TEX_TEXDESC_21                     (0x735)
#define LWE397_TEX_TEXDESC_21_WIDTH                 31:20

#define LWE397_TEX_TEXDESC_21_HEIGHT                        19:8

#define LWE397_TEX_TEXDESC_21_NORMALIZE                     7:7
#define LWE397_TEX_TEXDESC_21_NORMALIZE_DISABLE                   (0)
#define LWE397_TEX_TEXDESC_21_NORMALIZE_ENABLE                    (1)

#define LWE397_TEX_TEXDESC_21_LOG2_WIDTH                    31:28

#define LWE397_TEX_TEXDESC_21_LOG2_HEIGHT                   27:24

#define LWE397_TEX_TEXDESC_21_LOD_MIN                       23:16

#define LWE397_TEX_TEXDESC_21_LOD_MAX                       15:8

#define LWE397_TEX_TEXDESC_21_BASE_LEVEL_ONLY                       7:7
#define LWE397_TEX_TEXDESC_21_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_21_BASE_LEVEL_ONLY_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_21_NON_POWER_OF_TWO                      6:6
#define LWE397_TEX_TEXDESC_21_NON_POWER_OF_TWO_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_21_NON_POWER_OF_TWO_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_21_ARRAY_MAX                     5:0

#define LWE397_TEX_TEXDESC_21_TRILINEAR_OPT                 30:30
#define LWE397_TEX_TEXDESC_21_TRILINEAR_OPT_DISABLE                       (0)
#define LWE397_TEX_TEXDESC_21_TRILINEAR_OPT_ENABLE                        (1)

#define LWE397_TEX_TEXDESC_21_LERP_MAG                      29:29

#define LWE397_TEX_TEXDESC_21_LERP_MIN                      28:28

#define LWE397_TEX_TEXDESC_21_LERP_MIP                      27:27

#define LWE397_TEX_TEXDESC_21_LOD_BIAS                      26:18

#define LWE397_TEX_TEXDESC_21_MAX_ANISO                     17:14

#define LWE397_TEX_TEXDESC_21_SURF_FORMAT                   13:8

#define LWE397_TEX_TEXDESC_21_LWBEMAP                       7:7

#define LWE397_TEX_TEXDESC_21_LAYOUT                        6:4
#define LWE397_TEX_TEXDESC_21_LAYOUT_LINEAR                       (0)
#define LWE397_TEX_TEXDESC_21_LAYOUT_SWIZZLED                     (1)
#define LWE397_TEX_TEXDESC_21_LAYOUT_TILED_LINEAR                 (2)
#define LWE397_TEX_TEXDESC_21_LAYOUT_TILED_SWIZZLED                       (3)
#define LWE397_TEX_TEXDESC_21_LAYOUT_XY_TILED_LINEAR                      (4)
#define LWE397_TEX_TEXDESC_21_LAYOUT_XY_TILED_SWIZZLED                    (5)

#define LWE397_TEX_TEXDESC_21_MIRROR_S                      3:3
#define LWE397_TEX_TEXDESC_21_MIRROR_S_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_21_MIRROR_S_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_21_MIRROR_T                      2:2
#define LWE397_TEX_TEXDESC_21_MIRROR_T_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_21_MIRROR_T_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_21_CLAMP_S                       1:1
#define LWE397_TEX_TEXDESC_21_CLAMP_S_WRAP                        (0)
#define LWE397_TEX_TEXDESC_21_CLAMP_S_CLAMP                       (1)

#define LWE397_TEX_TEXDESC_21_CLAMP_T                       0:0
#define LWE397_TEX_TEXDESC_21_CLAMP_T_WRAP                        (0)
#define LWE397_TEX_TEXDESC_21_CLAMP_T_CLAMP                       (1)


// Register LWE397_TEX_TEXDESC_22  
#define LWE397_TEX_TEXDESC_22                     (0x736)
#define LWE397_TEX_TEXDESC_22_WIDTH                 31:20

#define LWE397_TEX_TEXDESC_22_HEIGHT                        19:8

#define LWE397_TEX_TEXDESC_22_NORMALIZE                     7:7
#define LWE397_TEX_TEXDESC_22_NORMALIZE_DISABLE                   (0)
#define LWE397_TEX_TEXDESC_22_NORMALIZE_ENABLE                    (1)

#define LWE397_TEX_TEXDESC_22_LOG2_WIDTH                    31:28

#define LWE397_TEX_TEXDESC_22_LOG2_HEIGHT                   27:24

#define LWE397_TEX_TEXDESC_22_LOD_MIN                       23:16

#define LWE397_TEX_TEXDESC_22_LOD_MAX                       15:8

#define LWE397_TEX_TEXDESC_22_BASE_LEVEL_ONLY                       7:7
#define LWE397_TEX_TEXDESC_22_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_22_BASE_LEVEL_ONLY_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_22_NON_POWER_OF_TWO                      6:6
#define LWE397_TEX_TEXDESC_22_NON_POWER_OF_TWO_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_22_NON_POWER_OF_TWO_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_22_ARRAY_MAX                     5:0

#define LWE397_TEX_TEXDESC_22_TRILINEAR_OPT                 30:30
#define LWE397_TEX_TEXDESC_22_TRILINEAR_OPT_DISABLE                       (0)
#define LWE397_TEX_TEXDESC_22_TRILINEAR_OPT_ENABLE                        (1)

#define LWE397_TEX_TEXDESC_22_LERP_MAG                      29:29

#define LWE397_TEX_TEXDESC_22_LERP_MIN                      28:28

#define LWE397_TEX_TEXDESC_22_LERP_MIP                      27:27

#define LWE397_TEX_TEXDESC_22_LOD_BIAS                      26:18

#define LWE397_TEX_TEXDESC_22_MAX_ANISO                     17:14

#define LWE397_TEX_TEXDESC_22_SURF_FORMAT                   13:8

#define LWE397_TEX_TEXDESC_22_LWBEMAP                       7:7

#define LWE397_TEX_TEXDESC_22_LAYOUT                        6:4
#define LWE397_TEX_TEXDESC_22_LAYOUT_LINEAR                       (0)
#define LWE397_TEX_TEXDESC_22_LAYOUT_SWIZZLED                     (1)
#define LWE397_TEX_TEXDESC_22_LAYOUT_TILED_LINEAR                 (2)
#define LWE397_TEX_TEXDESC_22_LAYOUT_TILED_SWIZZLED                       (3)
#define LWE397_TEX_TEXDESC_22_LAYOUT_XY_TILED_LINEAR                      (4)
#define LWE397_TEX_TEXDESC_22_LAYOUT_XY_TILED_SWIZZLED                    (5)

#define LWE397_TEX_TEXDESC_22_MIRROR_S                      3:3
#define LWE397_TEX_TEXDESC_22_MIRROR_S_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_22_MIRROR_S_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_22_MIRROR_T                      2:2
#define LWE397_TEX_TEXDESC_22_MIRROR_T_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_22_MIRROR_T_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_22_CLAMP_S                       1:1
#define LWE397_TEX_TEXDESC_22_CLAMP_S_WRAP                        (0)
#define LWE397_TEX_TEXDESC_22_CLAMP_S_CLAMP                       (1)

#define LWE397_TEX_TEXDESC_22_CLAMP_T                       0:0
#define LWE397_TEX_TEXDESC_22_CLAMP_T_WRAP                        (0)
#define LWE397_TEX_TEXDESC_22_CLAMP_T_CLAMP                       (1)


// Register LWE397_TEX_TEXDESC_23  
#define LWE397_TEX_TEXDESC_23                     (0x737)
#define LWE397_TEX_TEXDESC_23_WIDTH                 31:20

#define LWE397_TEX_TEXDESC_23_HEIGHT                        19:8

#define LWE397_TEX_TEXDESC_23_NORMALIZE                     7:7
#define LWE397_TEX_TEXDESC_23_NORMALIZE_DISABLE                   (0)
#define LWE397_TEX_TEXDESC_23_NORMALIZE_ENABLE                    (1)

#define LWE397_TEX_TEXDESC_23_LOG2_WIDTH                    31:28

#define LWE397_TEX_TEXDESC_23_LOG2_HEIGHT                   27:24

#define LWE397_TEX_TEXDESC_23_LOD_MIN                       23:16

#define LWE397_TEX_TEXDESC_23_LOD_MAX                       15:8

#define LWE397_TEX_TEXDESC_23_BASE_LEVEL_ONLY                       7:7
#define LWE397_TEX_TEXDESC_23_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_23_BASE_LEVEL_ONLY_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_23_NON_POWER_OF_TWO                      6:6
#define LWE397_TEX_TEXDESC_23_NON_POWER_OF_TWO_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_23_NON_POWER_OF_TWO_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_23_ARRAY_MAX                     5:0

#define LWE397_TEX_TEXDESC_23_TRILINEAR_OPT                 30:30
#define LWE397_TEX_TEXDESC_23_TRILINEAR_OPT_DISABLE                       (0)
#define LWE397_TEX_TEXDESC_23_TRILINEAR_OPT_ENABLE                        (1)

#define LWE397_TEX_TEXDESC_23_LERP_MAG                      29:29

#define LWE397_TEX_TEXDESC_23_LERP_MIN                      28:28

#define LWE397_TEX_TEXDESC_23_LERP_MIP                      27:27

#define LWE397_TEX_TEXDESC_23_LOD_BIAS                      26:18

#define LWE397_TEX_TEXDESC_23_MAX_ANISO                     17:14

#define LWE397_TEX_TEXDESC_23_SURF_FORMAT                   13:8

#define LWE397_TEX_TEXDESC_23_LWBEMAP                       7:7

#define LWE397_TEX_TEXDESC_23_LAYOUT                        6:4
#define LWE397_TEX_TEXDESC_23_LAYOUT_LINEAR                       (0)
#define LWE397_TEX_TEXDESC_23_LAYOUT_SWIZZLED                     (1)
#define LWE397_TEX_TEXDESC_23_LAYOUT_TILED_LINEAR                 (2)
#define LWE397_TEX_TEXDESC_23_LAYOUT_TILED_SWIZZLED                       (3)
#define LWE397_TEX_TEXDESC_23_LAYOUT_XY_TILED_LINEAR                      (4)
#define LWE397_TEX_TEXDESC_23_LAYOUT_XY_TILED_SWIZZLED                    (5)

#define LWE397_TEX_TEXDESC_23_MIRROR_S                      3:3
#define LWE397_TEX_TEXDESC_23_MIRROR_S_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_23_MIRROR_S_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_23_MIRROR_T                      2:2
#define LWE397_TEX_TEXDESC_23_MIRROR_T_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_23_MIRROR_T_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_23_CLAMP_S                       1:1
#define LWE397_TEX_TEXDESC_23_CLAMP_S_WRAP                        (0)
#define LWE397_TEX_TEXDESC_23_CLAMP_S_CLAMP                       (1)

#define LWE397_TEX_TEXDESC_23_CLAMP_T                       0:0
#define LWE397_TEX_TEXDESC_23_CLAMP_T_WRAP                        (0)
#define LWE397_TEX_TEXDESC_23_CLAMP_T_CLAMP                       (1)


// Register LWE397_TEX_TEXDESC_24  
#define LWE397_TEX_TEXDESC_24                     (0x738)
#define LWE397_TEX_TEXDESC_24_WIDTH                 31:20

#define LWE397_TEX_TEXDESC_24_HEIGHT                        19:8

#define LWE397_TEX_TEXDESC_24_NORMALIZE                     7:7
#define LWE397_TEX_TEXDESC_24_NORMALIZE_DISABLE                   (0)
#define LWE397_TEX_TEXDESC_24_NORMALIZE_ENABLE                    (1)

#define LWE397_TEX_TEXDESC_24_LOG2_WIDTH                    31:28

#define LWE397_TEX_TEXDESC_24_LOG2_HEIGHT                   27:24

#define LWE397_TEX_TEXDESC_24_LOD_MIN                       23:16

#define LWE397_TEX_TEXDESC_24_LOD_MAX                       15:8

#define LWE397_TEX_TEXDESC_24_BASE_LEVEL_ONLY                       7:7
#define LWE397_TEX_TEXDESC_24_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_24_BASE_LEVEL_ONLY_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_24_NON_POWER_OF_TWO                      6:6
#define LWE397_TEX_TEXDESC_24_NON_POWER_OF_TWO_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_24_NON_POWER_OF_TWO_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_24_ARRAY_MAX                     5:0

#define LWE397_TEX_TEXDESC_24_TRILINEAR_OPT                 30:30
#define LWE397_TEX_TEXDESC_24_TRILINEAR_OPT_DISABLE                       (0)
#define LWE397_TEX_TEXDESC_24_TRILINEAR_OPT_ENABLE                        (1)

#define LWE397_TEX_TEXDESC_24_LERP_MAG                      29:29

#define LWE397_TEX_TEXDESC_24_LERP_MIN                      28:28

#define LWE397_TEX_TEXDESC_24_LERP_MIP                      27:27

#define LWE397_TEX_TEXDESC_24_LOD_BIAS                      26:18

#define LWE397_TEX_TEXDESC_24_MAX_ANISO                     17:14

#define LWE397_TEX_TEXDESC_24_SURF_FORMAT                   13:8

#define LWE397_TEX_TEXDESC_24_LWBEMAP                       7:7

#define LWE397_TEX_TEXDESC_24_LAYOUT                        6:4
#define LWE397_TEX_TEXDESC_24_LAYOUT_LINEAR                       (0)
#define LWE397_TEX_TEXDESC_24_LAYOUT_SWIZZLED                     (1)
#define LWE397_TEX_TEXDESC_24_LAYOUT_TILED_LINEAR                 (2)
#define LWE397_TEX_TEXDESC_24_LAYOUT_TILED_SWIZZLED                       (3)
#define LWE397_TEX_TEXDESC_24_LAYOUT_XY_TILED_LINEAR                      (4)
#define LWE397_TEX_TEXDESC_24_LAYOUT_XY_TILED_SWIZZLED                    (5)

#define LWE397_TEX_TEXDESC_24_MIRROR_S                      3:3
#define LWE397_TEX_TEXDESC_24_MIRROR_S_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_24_MIRROR_S_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_24_MIRROR_T                      2:2
#define LWE397_TEX_TEXDESC_24_MIRROR_T_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_24_MIRROR_T_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_24_CLAMP_S                       1:1
#define LWE397_TEX_TEXDESC_24_CLAMP_S_WRAP                        (0)
#define LWE397_TEX_TEXDESC_24_CLAMP_S_CLAMP                       (1)

#define LWE397_TEX_TEXDESC_24_CLAMP_T                       0:0
#define LWE397_TEX_TEXDESC_24_CLAMP_T_WRAP                        (0)
#define LWE397_TEX_TEXDESC_24_CLAMP_T_CLAMP                       (1)


// Register LWE397_TEX_TEXDESC_25  
#define LWE397_TEX_TEXDESC_25                     (0x739)
#define LWE397_TEX_TEXDESC_25_WIDTH                 31:20

#define LWE397_TEX_TEXDESC_25_HEIGHT                        19:8

#define LWE397_TEX_TEXDESC_25_NORMALIZE                     7:7
#define LWE397_TEX_TEXDESC_25_NORMALIZE_DISABLE                   (0)
#define LWE397_TEX_TEXDESC_25_NORMALIZE_ENABLE                    (1)

#define LWE397_TEX_TEXDESC_25_LOG2_WIDTH                    31:28

#define LWE397_TEX_TEXDESC_25_LOG2_HEIGHT                   27:24

#define LWE397_TEX_TEXDESC_25_LOD_MIN                       23:16

#define LWE397_TEX_TEXDESC_25_LOD_MAX                       15:8

#define LWE397_TEX_TEXDESC_25_BASE_LEVEL_ONLY                       7:7
#define LWE397_TEX_TEXDESC_25_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_25_BASE_LEVEL_ONLY_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_25_NON_POWER_OF_TWO                      6:6
#define LWE397_TEX_TEXDESC_25_NON_POWER_OF_TWO_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_25_NON_POWER_OF_TWO_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_25_ARRAY_MAX                     5:0

#define LWE397_TEX_TEXDESC_25_TRILINEAR_OPT                 30:30
#define LWE397_TEX_TEXDESC_25_TRILINEAR_OPT_DISABLE                       (0)
#define LWE397_TEX_TEXDESC_25_TRILINEAR_OPT_ENABLE                        (1)

#define LWE397_TEX_TEXDESC_25_LERP_MAG                      29:29

#define LWE397_TEX_TEXDESC_25_LERP_MIN                      28:28

#define LWE397_TEX_TEXDESC_25_LERP_MIP                      27:27

#define LWE397_TEX_TEXDESC_25_LOD_BIAS                      26:18

#define LWE397_TEX_TEXDESC_25_MAX_ANISO                     17:14

#define LWE397_TEX_TEXDESC_25_SURF_FORMAT                   13:8

#define LWE397_TEX_TEXDESC_25_LWBEMAP                       7:7

#define LWE397_TEX_TEXDESC_25_LAYOUT                        6:4
#define LWE397_TEX_TEXDESC_25_LAYOUT_LINEAR                       (0)
#define LWE397_TEX_TEXDESC_25_LAYOUT_SWIZZLED                     (1)
#define LWE397_TEX_TEXDESC_25_LAYOUT_TILED_LINEAR                 (2)
#define LWE397_TEX_TEXDESC_25_LAYOUT_TILED_SWIZZLED                       (3)
#define LWE397_TEX_TEXDESC_25_LAYOUT_XY_TILED_LINEAR                      (4)
#define LWE397_TEX_TEXDESC_25_LAYOUT_XY_TILED_SWIZZLED                    (5)

#define LWE397_TEX_TEXDESC_25_MIRROR_S                      3:3
#define LWE397_TEX_TEXDESC_25_MIRROR_S_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_25_MIRROR_S_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_25_MIRROR_T                      2:2
#define LWE397_TEX_TEXDESC_25_MIRROR_T_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_25_MIRROR_T_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_25_CLAMP_S                       1:1
#define LWE397_TEX_TEXDESC_25_CLAMP_S_WRAP                        (0)
#define LWE397_TEX_TEXDESC_25_CLAMP_S_CLAMP                       (1)

#define LWE397_TEX_TEXDESC_25_CLAMP_T                       0:0
#define LWE397_TEX_TEXDESC_25_CLAMP_T_WRAP                        (0)
#define LWE397_TEX_TEXDESC_25_CLAMP_T_CLAMP                       (1)


// Register LWE397_TEX_TEXDESC_26  
#define LWE397_TEX_TEXDESC_26                     (0x73a)
#define LWE397_TEX_TEXDESC_26_WIDTH                 31:20

#define LWE397_TEX_TEXDESC_26_HEIGHT                        19:8

#define LWE397_TEX_TEXDESC_26_NORMALIZE                     7:7
#define LWE397_TEX_TEXDESC_26_NORMALIZE_DISABLE                   (0)
#define LWE397_TEX_TEXDESC_26_NORMALIZE_ENABLE                    (1)

#define LWE397_TEX_TEXDESC_26_LOG2_WIDTH                    31:28

#define LWE397_TEX_TEXDESC_26_LOG2_HEIGHT                   27:24

#define LWE397_TEX_TEXDESC_26_LOD_MIN                       23:16

#define LWE397_TEX_TEXDESC_26_LOD_MAX                       15:8

#define LWE397_TEX_TEXDESC_26_BASE_LEVEL_ONLY                       7:7
#define LWE397_TEX_TEXDESC_26_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_26_BASE_LEVEL_ONLY_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_26_NON_POWER_OF_TWO                      6:6
#define LWE397_TEX_TEXDESC_26_NON_POWER_OF_TWO_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_26_NON_POWER_OF_TWO_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_26_ARRAY_MAX                     5:0

#define LWE397_TEX_TEXDESC_26_TRILINEAR_OPT                 30:30
#define LWE397_TEX_TEXDESC_26_TRILINEAR_OPT_DISABLE                       (0)
#define LWE397_TEX_TEXDESC_26_TRILINEAR_OPT_ENABLE                        (1)

#define LWE397_TEX_TEXDESC_26_LERP_MAG                      29:29

#define LWE397_TEX_TEXDESC_26_LERP_MIN                      28:28

#define LWE397_TEX_TEXDESC_26_LERP_MIP                      27:27

#define LWE397_TEX_TEXDESC_26_LOD_BIAS                      26:18

#define LWE397_TEX_TEXDESC_26_MAX_ANISO                     17:14

#define LWE397_TEX_TEXDESC_26_SURF_FORMAT                   13:8

#define LWE397_TEX_TEXDESC_26_LWBEMAP                       7:7

#define LWE397_TEX_TEXDESC_26_LAYOUT                        6:4
#define LWE397_TEX_TEXDESC_26_LAYOUT_LINEAR                       (0)
#define LWE397_TEX_TEXDESC_26_LAYOUT_SWIZZLED                     (1)
#define LWE397_TEX_TEXDESC_26_LAYOUT_TILED_LINEAR                 (2)
#define LWE397_TEX_TEXDESC_26_LAYOUT_TILED_SWIZZLED                       (3)
#define LWE397_TEX_TEXDESC_26_LAYOUT_XY_TILED_LINEAR                      (4)
#define LWE397_TEX_TEXDESC_26_LAYOUT_XY_TILED_SWIZZLED                    (5)

#define LWE397_TEX_TEXDESC_26_MIRROR_S                      3:3
#define LWE397_TEX_TEXDESC_26_MIRROR_S_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_26_MIRROR_S_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_26_MIRROR_T                      2:2
#define LWE397_TEX_TEXDESC_26_MIRROR_T_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_26_MIRROR_T_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_26_CLAMP_S                       1:1
#define LWE397_TEX_TEXDESC_26_CLAMP_S_WRAP                        (0)
#define LWE397_TEX_TEXDESC_26_CLAMP_S_CLAMP                       (1)

#define LWE397_TEX_TEXDESC_26_CLAMP_T                       0:0
#define LWE397_TEX_TEXDESC_26_CLAMP_T_WRAP                        (0)
#define LWE397_TEX_TEXDESC_26_CLAMP_T_CLAMP                       (1)


// Register LWE397_TEX_TEXDESC_27  
#define LWE397_TEX_TEXDESC_27                     (0x73b)
#define LWE397_TEX_TEXDESC_27_WIDTH                 31:20

#define LWE397_TEX_TEXDESC_27_HEIGHT                        19:8

#define LWE397_TEX_TEXDESC_27_NORMALIZE                     7:7
#define LWE397_TEX_TEXDESC_27_NORMALIZE_DISABLE                   (0)
#define LWE397_TEX_TEXDESC_27_NORMALIZE_ENABLE                    (1)

#define LWE397_TEX_TEXDESC_27_LOG2_WIDTH                    31:28

#define LWE397_TEX_TEXDESC_27_LOG2_HEIGHT                   27:24

#define LWE397_TEX_TEXDESC_27_LOD_MIN                       23:16

#define LWE397_TEX_TEXDESC_27_LOD_MAX                       15:8

#define LWE397_TEX_TEXDESC_27_BASE_LEVEL_ONLY                       7:7
#define LWE397_TEX_TEXDESC_27_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_27_BASE_LEVEL_ONLY_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_27_NON_POWER_OF_TWO                      6:6
#define LWE397_TEX_TEXDESC_27_NON_POWER_OF_TWO_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_27_NON_POWER_OF_TWO_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_27_ARRAY_MAX                     5:0

#define LWE397_TEX_TEXDESC_27_TRILINEAR_OPT                 30:30
#define LWE397_TEX_TEXDESC_27_TRILINEAR_OPT_DISABLE                       (0)
#define LWE397_TEX_TEXDESC_27_TRILINEAR_OPT_ENABLE                        (1)

#define LWE397_TEX_TEXDESC_27_LERP_MAG                      29:29

#define LWE397_TEX_TEXDESC_27_LERP_MIN                      28:28

#define LWE397_TEX_TEXDESC_27_LERP_MIP                      27:27

#define LWE397_TEX_TEXDESC_27_LOD_BIAS                      26:18

#define LWE397_TEX_TEXDESC_27_MAX_ANISO                     17:14

#define LWE397_TEX_TEXDESC_27_SURF_FORMAT                   13:8

#define LWE397_TEX_TEXDESC_27_LWBEMAP                       7:7

#define LWE397_TEX_TEXDESC_27_LAYOUT                        6:4
#define LWE397_TEX_TEXDESC_27_LAYOUT_LINEAR                       (0)
#define LWE397_TEX_TEXDESC_27_LAYOUT_SWIZZLED                     (1)
#define LWE397_TEX_TEXDESC_27_LAYOUT_TILED_LINEAR                 (2)
#define LWE397_TEX_TEXDESC_27_LAYOUT_TILED_SWIZZLED                       (3)
#define LWE397_TEX_TEXDESC_27_LAYOUT_XY_TILED_LINEAR                      (4)
#define LWE397_TEX_TEXDESC_27_LAYOUT_XY_TILED_SWIZZLED                    (5)

#define LWE397_TEX_TEXDESC_27_MIRROR_S                      3:3
#define LWE397_TEX_TEXDESC_27_MIRROR_S_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_27_MIRROR_S_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_27_MIRROR_T                      2:2
#define LWE397_TEX_TEXDESC_27_MIRROR_T_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_27_MIRROR_T_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_27_CLAMP_S                       1:1
#define LWE397_TEX_TEXDESC_27_CLAMP_S_WRAP                        (0)
#define LWE397_TEX_TEXDESC_27_CLAMP_S_CLAMP                       (1)

#define LWE397_TEX_TEXDESC_27_CLAMP_T                       0:0
#define LWE397_TEX_TEXDESC_27_CLAMP_T_WRAP                        (0)
#define LWE397_TEX_TEXDESC_27_CLAMP_T_CLAMP                       (1)


// Register LWE397_TEX_TEXDESC_28  
#define LWE397_TEX_TEXDESC_28                     (0x73c)
#define LWE397_TEX_TEXDESC_28_WIDTH                 31:20

#define LWE397_TEX_TEXDESC_28_HEIGHT                        19:8

#define LWE397_TEX_TEXDESC_28_NORMALIZE                     7:7
#define LWE397_TEX_TEXDESC_28_NORMALIZE_DISABLE                   (0)
#define LWE397_TEX_TEXDESC_28_NORMALIZE_ENABLE                    (1)

#define LWE397_TEX_TEXDESC_28_LOG2_WIDTH                    31:28

#define LWE397_TEX_TEXDESC_28_LOG2_HEIGHT                   27:24

#define LWE397_TEX_TEXDESC_28_LOD_MIN                       23:16

#define LWE397_TEX_TEXDESC_28_LOD_MAX                       15:8

#define LWE397_TEX_TEXDESC_28_BASE_LEVEL_ONLY                       7:7
#define LWE397_TEX_TEXDESC_28_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_28_BASE_LEVEL_ONLY_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_28_NON_POWER_OF_TWO                      6:6
#define LWE397_TEX_TEXDESC_28_NON_POWER_OF_TWO_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_28_NON_POWER_OF_TWO_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_28_ARRAY_MAX                     5:0

#define LWE397_TEX_TEXDESC_28_TRILINEAR_OPT                 30:30
#define LWE397_TEX_TEXDESC_28_TRILINEAR_OPT_DISABLE                       (0)
#define LWE397_TEX_TEXDESC_28_TRILINEAR_OPT_ENABLE                        (1)

#define LWE397_TEX_TEXDESC_28_LERP_MAG                      29:29

#define LWE397_TEX_TEXDESC_28_LERP_MIN                      28:28

#define LWE397_TEX_TEXDESC_28_LERP_MIP                      27:27

#define LWE397_TEX_TEXDESC_28_LOD_BIAS                      26:18

#define LWE397_TEX_TEXDESC_28_MAX_ANISO                     17:14

#define LWE397_TEX_TEXDESC_28_SURF_FORMAT                   13:8

#define LWE397_TEX_TEXDESC_28_LWBEMAP                       7:7

#define LWE397_TEX_TEXDESC_28_LAYOUT                        6:4
#define LWE397_TEX_TEXDESC_28_LAYOUT_LINEAR                       (0)
#define LWE397_TEX_TEXDESC_28_LAYOUT_SWIZZLED                     (1)
#define LWE397_TEX_TEXDESC_28_LAYOUT_TILED_LINEAR                 (2)
#define LWE397_TEX_TEXDESC_28_LAYOUT_TILED_SWIZZLED                       (3)
#define LWE397_TEX_TEXDESC_28_LAYOUT_XY_TILED_LINEAR                      (4)
#define LWE397_TEX_TEXDESC_28_LAYOUT_XY_TILED_SWIZZLED                    (5)

#define LWE397_TEX_TEXDESC_28_MIRROR_S                      3:3
#define LWE397_TEX_TEXDESC_28_MIRROR_S_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_28_MIRROR_S_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_28_MIRROR_T                      2:2
#define LWE397_TEX_TEXDESC_28_MIRROR_T_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_28_MIRROR_T_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_28_CLAMP_S                       1:1
#define LWE397_TEX_TEXDESC_28_CLAMP_S_WRAP                        (0)
#define LWE397_TEX_TEXDESC_28_CLAMP_S_CLAMP                       (1)

#define LWE397_TEX_TEXDESC_28_CLAMP_T                       0:0
#define LWE397_TEX_TEXDESC_28_CLAMP_T_WRAP                        (0)
#define LWE397_TEX_TEXDESC_28_CLAMP_T_CLAMP                       (1)


// Register LWE397_TEX_TEXDESC_29  
#define LWE397_TEX_TEXDESC_29                     (0x73d)
#define LWE397_TEX_TEXDESC_29_WIDTH                 31:20

#define LWE397_TEX_TEXDESC_29_HEIGHT                        19:8

#define LWE397_TEX_TEXDESC_29_NORMALIZE                     7:7
#define LWE397_TEX_TEXDESC_29_NORMALIZE_DISABLE                   (0)
#define LWE397_TEX_TEXDESC_29_NORMALIZE_ENABLE                    (1)

#define LWE397_TEX_TEXDESC_29_LOG2_WIDTH                    31:28

#define LWE397_TEX_TEXDESC_29_LOG2_HEIGHT                   27:24

#define LWE397_TEX_TEXDESC_29_LOD_MIN                       23:16

#define LWE397_TEX_TEXDESC_29_LOD_MAX                       15:8

#define LWE397_TEX_TEXDESC_29_BASE_LEVEL_ONLY                       7:7
#define LWE397_TEX_TEXDESC_29_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_29_BASE_LEVEL_ONLY_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_29_NON_POWER_OF_TWO                      6:6
#define LWE397_TEX_TEXDESC_29_NON_POWER_OF_TWO_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_29_NON_POWER_OF_TWO_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_29_ARRAY_MAX                     5:0

#define LWE397_TEX_TEXDESC_29_TRILINEAR_OPT                 30:30
#define LWE397_TEX_TEXDESC_29_TRILINEAR_OPT_DISABLE                       (0)
#define LWE397_TEX_TEXDESC_29_TRILINEAR_OPT_ENABLE                        (1)

#define LWE397_TEX_TEXDESC_29_LERP_MAG                      29:29

#define LWE397_TEX_TEXDESC_29_LERP_MIN                      28:28

#define LWE397_TEX_TEXDESC_29_LERP_MIP                      27:27

#define LWE397_TEX_TEXDESC_29_LOD_BIAS                      26:18

#define LWE397_TEX_TEXDESC_29_MAX_ANISO                     17:14

#define LWE397_TEX_TEXDESC_29_SURF_FORMAT                   13:8

#define LWE397_TEX_TEXDESC_29_LWBEMAP                       7:7

#define LWE397_TEX_TEXDESC_29_LAYOUT                        6:4
#define LWE397_TEX_TEXDESC_29_LAYOUT_LINEAR                       (0)
#define LWE397_TEX_TEXDESC_29_LAYOUT_SWIZZLED                     (1)
#define LWE397_TEX_TEXDESC_29_LAYOUT_TILED_LINEAR                 (2)
#define LWE397_TEX_TEXDESC_29_LAYOUT_TILED_SWIZZLED                       (3)
#define LWE397_TEX_TEXDESC_29_LAYOUT_XY_TILED_LINEAR                      (4)
#define LWE397_TEX_TEXDESC_29_LAYOUT_XY_TILED_SWIZZLED                    (5)

#define LWE397_TEX_TEXDESC_29_MIRROR_S                      3:3
#define LWE397_TEX_TEXDESC_29_MIRROR_S_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_29_MIRROR_S_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_29_MIRROR_T                      2:2
#define LWE397_TEX_TEXDESC_29_MIRROR_T_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_29_MIRROR_T_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_29_CLAMP_S                       1:1
#define LWE397_TEX_TEXDESC_29_CLAMP_S_WRAP                        (0)
#define LWE397_TEX_TEXDESC_29_CLAMP_S_CLAMP                       (1)

#define LWE397_TEX_TEXDESC_29_CLAMP_T                       0:0
#define LWE397_TEX_TEXDESC_29_CLAMP_T_WRAP                        (0)
#define LWE397_TEX_TEXDESC_29_CLAMP_T_CLAMP                       (1)


// Register LWE397_TEX_TEXDESC_30  
#define LWE397_TEX_TEXDESC_30                     (0x73e)
#define LWE397_TEX_TEXDESC_30_WIDTH                 31:20

#define LWE397_TEX_TEXDESC_30_HEIGHT                        19:8

#define LWE397_TEX_TEXDESC_30_NORMALIZE                     7:7
#define LWE397_TEX_TEXDESC_30_NORMALIZE_DISABLE                   (0)
#define LWE397_TEX_TEXDESC_30_NORMALIZE_ENABLE                    (1)

#define LWE397_TEX_TEXDESC_30_LOG2_WIDTH                    31:28

#define LWE397_TEX_TEXDESC_30_LOG2_HEIGHT                   27:24

#define LWE397_TEX_TEXDESC_30_LOD_MIN                       23:16

#define LWE397_TEX_TEXDESC_30_LOD_MAX                       15:8

#define LWE397_TEX_TEXDESC_30_BASE_LEVEL_ONLY                       7:7
#define LWE397_TEX_TEXDESC_30_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_30_BASE_LEVEL_ONLY_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_30_NON_POWER_OF_TWO                      6:6
#define LWE397_TEX_TEXDESC_30_NON_POWER_OF_TWO_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_30_NON_POWER_OF_TWO_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_30_ARRAY_MAX                     5:0

#define LWE397_TEX_TEXDESC_30_TRILINEAR_OPT                 30:30
#define LWE397_TEX_TEXDESC_30_TRILINEAR_OPT_DISABLE                       (0)
#define LWE397_TEX_TEXDESC_30_TRILINEAR_OPT_ENABLE                        (1)

#define LWE397_TEX_TEXDESC_30_LERP_MAG                      29:29

#define LWE397_TEX_TEXDESC_30_LERP_MIN                      28:28

#define LWE397_TEX_TEXDESC_30_LERP_MIP                      27:27

#define LWE397_TEX_TEXDESC_30_LOD_BIAS                      26:18

#define LWE397_TEX_TEXDESC_30_MAX_ANISO                     17:14

#define LWE397_TEX_TEXDESC_30_SURF_FORMAT                   13:8

#define LWE397_TEX_TEXDESC_30_LWBEMAP                       7:7

#define LWE397_TEX_TEXDESC_30_LAYOUT                        6:4
#define LWE397_TEX_TEXDESC_30_LAYOUT_LINEAR                       (0)
#define LWE397_TEX_TEXDESC_30_LAYOUT_SWIZZLED                     (1)
#define LWE397_TEX_TEXDESC_30_LAYOUT_TILED_LINEAR                 (2)
#define LWE397_TEX_TEXDESC_30_LAYOUT_TILED_SWIZZLED                       (3)
#define LWE397_TEX_TEXDESC_30_LAYOUT_XY_TILED_LINEAR                      (4)
#define LWE397_TEX_TEXDESC_30_LAYOUT_XY_TILED_SWIZZLED                    (5)

#define LWE397_TEX_TEXDESC_30_MIRROR_S                      3:3
#define LWE397_TEX_TEXDESC_30_MIRROR_S_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_30_MIRROR_S_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_30_MIRROR_T                      2:2
#define LWE397_TEX_TEXDESC_30_MIRROR_T_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_30_MIRROR_T_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_30_CLAMP_S                       1:1
#define LWE397_TEX_TEXDESC_30_CLAMP_S_WRAP                        (0)
#define LWE397_TEX_TEXDESC_30_CLAMP_S_CLAMP                       (1)

#define LWE397_TEX_TEXDESC_30_CLAMP_T                       0:0
#define LWE397_TEX_TEXDESC_30_CLAMP_T_WRAP                        (0)
#define LWE397_TEX_TEXDESC_30_CLAMP_T_CLAMP                       (1)


// Register LWE397_TEX_TEXDESC_31  
#define LWE397_TEX_TEXDESC_31                     (0x73f)
#define LWE397_TEX_TEXDESC_31_WIDTH                 31:20

#define LWE397_TEX_TEXDESC_31_HEIGHT                        19:8

#define LWE397_TEX_TEXDESC_31_NORMALIZE                     7:7
#define LWE397_TEX_TEXDESC_31_NORMALIZE_DISABLE                   (0)
#define LWE397_TEX_TEXDESC_31_NORMALIZE_ENABLE                    (1)

#define LWE397_TEX_TEXDESC_31_LOG2_WIDTH                    31:28

#define LWE397_TEX_TEXDESC_31_LOG2_HEIGHT                   27:24

#define LWE397_TEX_TEXDESC_31_LOD_MIN                       23:16

#define LWE397_TEX_TEXDESC_31_LOD_MAX                       15:8

#define LWE397_TEX_TEXDESC_31_BASE_LEVEL_ONLY                       7:7
#define LWE397_TEX_TEXDESC_31_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_31_BASE_LEVEL_ONLY_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_31_NON_POWER_OF_TWO                      6:6
#define LWE397_TEX_TEXDESC_31_NON_POWER_OF_TWO_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_31_NON_POWER_OF_TWO_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_31_ARRAY_MAX                     5:0

#define LWE397_TEX_TEXDESC_31_TRILINEAR_OPT                 30:30
#define LWE397_TEX_TEXDESC_31_TRILINEAR_OPT_DISABLE                       (0)
#define LWE397_TEX_TEXDESC_31_TRILINEAR_OPT_ENABLE                        (1)

#define LWE397_TEX_TEXDESC_31_LERP_MAG                      29:29

#define LWE397_TEX_TEXDESC_31_LERP_MIN                      28:28

#define LWE397_TEX_TEXDESC_31_LERP_MIP                      27:27

#define LWE397_TEX_TEXDESC_31_LOD_BIAS                      26:18

#define LWE397_TEX_TEXDESC_31_MAX_ANISO                     17:14

#define LWE397_TEX_TEXDESC_31_SURF_FORMAT                   13:8

#define LWE397_TEX_TEXDESC_31_LWBEMAP                       7:7

#define LWE397_TEX_TEXDESC_31_LAYOUT                        6:4
#define LWE397_TEX_TEXDESC_31_LAYOUT_LINEAR                       (0)
#define LWE397_TEX_TEXDESC_31_LAYOUT_SWIZZLED                     (1)
#define LWE397_TEX_TEXDESC_31_LAYOUT_TILED_LINEAR                 (2)
#define LWE397_TEX_TEXDESC_31_LAYOUT_TILED_SWIZZLED                       (3)
#define LWE397_TEX_TEXDESC_31_LAYOUT_XY_TILED_LINEAR                      (4)
#define LWE397_TEX_TEXDESC_31_LAYOUT_XY_TILED_SWIZZLED                    (5)

#define LWE397_TEX_TEXDESC_31_MIRROR_S                      3:3
#define LWE397_TEX_TEXDESC_31_MIRROR_S_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_31_MIRROR_S_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_31_MIRROR_T                      2:2
#define LWE397_TEX_TEXDESC_31_MIRROR_T_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_31_MIRROR_T_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_31_CLAMP_S                       1:1
#define LWE397_TEX_TEXDESC_31_CLAMP_S_WRAP                        (0)
#define LWE397_TEX_TEXDESC_31_CLAMP_S_CLAMP                       (1)

#define LWE397_TEX_TEXDESC_31_CLAMP_T                       0:0
#define LWE397_TEX_TEXDESC_31_CLAMP_T_WRAP                        (0)
#define LWE397_TEX_TEXDESC_31_CLAMP_T_CLAMP                       (1)


// Register LWE397_TEX_TEXCTL_0  
#define LWE397_TEX_TEXCTL_0                       (0x740)
#define LWE397_TEX_TEXCTL_0_ANISO_BIAS                      6:4

#define LWE397_TEX_TEXCTL_0_ANISO_OPT                       3:2
#define LWE397_TEX_TEXCTL_0_ANISO_OPT_DISABLE                     (0)
#define LWE397_TEX_TEXCTL_0_ANISO_OPT_SLOPE_2                     (1)
#define LWE397_TEX_TEXCTL_0_ANISO_OPT_FLOOR                       (2)
#define LWE397_TEX_TEXCTL_0_ANISO_OPT_FLOOR_EVEN                  (3)

#define LWE397_TEX_TEXCTL_0_TEXTURE_CACHE_EN                        0:0
#define LWE397_TEX_TEXCTL_0_TEXTURE_CACHE_EN_DISABLE                      (0)
#define LWE397_TEX_TEXCTL_0_TEXTURE_CACHE_EN_ENABLE                       (1)


// Register LWE397_TEX_CLKEN_OVERRIDE_0  
#define LWE397_TEX_CLKEN_OVERRIDE_0                       (0x741)
#define LWE397_TEX_CLKEN_OVERRIDE_0_TEXPBCLK_CLKEN_OVR                      0:0
#define LWE397_TEX_CLKEN_OVERRIDE_0_TEXPBCLK_CLKEN_OVR_CLK_GATED                  (0)
#define LWE397_TEX_CLKEN_OVERRIDE_0_TEXPBCLK_CLKEN_OVR_CLK_ALWAYS_ON                      (1)

#define LWE397_TEX_CLKEN_OVERRIDE_0_TEXPPBCLK_CLKEN_OVR                     1:1
#define LWE397_TEX_CLKEN_OVERRIDE_0_TEXPPBCLK_CLKEN_OVR_CLK_GATED                 (0)
#define LWE397_TEX_CLKEN_OVERRIDE_0_TEXPPBCLK_CLKEN_OVR_CLK_ALWAYS_ON                     (1)

#define LWE397_TEX_CLKEN_OVERRIDE_0_TEXLODCLK_CLKEN_OVR                     2:2
#define LWE397_TEX_CLKEN_OVERRIDE_0_TEXLODCLK_CLKEN_OVR_CLK_GATED                 (0)
#define LWE397_TEX_CLKEN_OVERRIDE_0_TEXLODCLK_CLKEN_OVR_CLK_ALWAYS_ON                     (1)

#define LWE397_TEX_CLKEN_OVERRIDE_0_TEXFORMATCLK_CLKEN_OVR                  3:3
#define LWE397_TEX_CLKEN_OVERRIDE_0_TEXFORMATCLK_CLKEN_OVR_CLK_GATED                      (0)
#define LWE397_TEX_CLKEN_OVERRIDE_0_TEXFORMATCLK_CLKEN_OVR_CLK_ALWAYS_ON                  (1)

#define LWE397_TEX_CLKEN_OVERRIDE_0_TEXLODLOWERDPCLK_CLKEN_OVR                      4:4
#define LWE397_TEX_CLKEN_OVERRIDE_0_TEXLODLOWERDPCLK_CLKEN_OVR_CLK_GATED                  (0)
#define LWE397_TEX_CLKEN_OVERRIDE_0_TEXLODLOWERDPCLK_CLKEN_OVR_CLK_ALWAYS_ON                      (1)

#define LWE397_TEX_CLKEN_OVERRIDE_0_TEXNONBYPCLK_CLKEN_OVR                  5:5
#define LWE397_TEX_CLKEN_OVERRIDE_0_TEXNONBYPCLK_CLKEN_OVR_CLK_GATED                      (0)
#define LWE397_TEX_CLKEN_OVERRIDE_0_TEXNONBYPCLK_CLKEN_OVR_CLK_ALWAYS_ON                  (1)

#define LWE397_TEX_CLKEN_OVERRIDE_0_TEXCACHECLK_CLKEN_OVR                   6:6
#define LWE397_TEX_CLKEN_OVERRIDE_0_TEXCACHECLK_CLKEN_OVR_CLK_GATED                       (0)
#define LWE397_TEX_CLKEN_OVERRIDE_0_TEXCACHECLK_CLKEN_OVR_CLK_ALWAYS_ON                   (1)


// Register LWE397_TEX_LW_MCCIF_FIFOCTRL_RO_0  
#define LWE397_TEX_LW_MCCIF_FIFOCTRL_RO_0                 (0x742)
#define LWE397_TEX_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_WRCL_MCLE2X                      0:0
#define LWE397_TEX_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_WRCL_MCLE2X_DISABLE                    (0)
#define LWE397_TEX_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_WRCL_MCLE2X_ENABLE                     (1)

#define LWE397_TEX_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_RDMC_RDFAST                      1:1
#define LWE397_TEX_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_RDMC_RDFAST_DISABLE                    (0)
#define LWE397_TEX_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_RDMC_RDFAST_ENABLE                     (1)

#define LWE397_TEX_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_WRMC_CLLE2X                      2:2
#define LWE397_TEX_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_WRMC_CLLE2X_DISABLE                    (0)
#define LWE397_TEX_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_WRMC_CLLE2X_ENABLE                     (1)

#define LWE397_TEX_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_RDCL_RDFAST                      3:3
#define LWE397_TEX_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_RDCL_RDFAST_DISABLE                    (0)
#define LWE397_TEX_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_RDCL_RDFAST_ENABLE                     (1)

#define LWE397_TEX_LW_MCCIF_FIFOCTRL_RO_0_LW_WCLK_OVERRIDE                  16:16

#define LWE397_TEX_LW_MCCIF_FIFOCTRL_RO_0_LW_RCLK_OVERRIDE                  17:17


#define LWE397_TEX_TEXDESC_NPOT_AUX(i)                              (0x750 + (i))
#define LWE397_TEX_TEXDESC_NPOT_AUX_BASE_LEVEL_ONLY                 24:24
#define LWE397_TEX_TEXDESC_NPOT_AUX_BASE_LEVEL_ONLY_DISABLE         0
#define LWE397_TEX_TEXDESC_NPOT_AUX_BASE_LEVEL_ONLY_ENABLE          1
#define LWE397_TEX_TEXDESC_NPOT_AUX_LOG2_WIDTH                      23:20
#define LWE397_TEX_TEXDESC_NPOT_AUX_LOG2_HEIGHT                     19:16
#define LWE397_TEX_TEXDESC_NPOT_AUX_LOD_MIN                         15:8
#define LWE397_TEX_TEXDESC_NPOT_AUX_LOD_MAX                         7:0

// Register LWE397_TEX_TEXDESC_NPOT_AUX_0  
#define LWE397_TEX_TEXDESC_NPOT_AUX_0                               (0x750)
#define LWE397_TEX_TEXDESC_NPOT_AUX_0_BASE_LEVEL_ONLY               24:24
#define LWE397_TEX_TEXDESC_NPOT_AUX_0_BASE_LEVEL_ONLY_DISABLE       (0)
#define LWE397_TEX_TEXDESC_NPOT_AUX_0_BASE_LEVEL_ONLY_ENABLE        (1)
#define LWE397_TEX_TEXDESC_NPOT_AUX_0_LOG2_WIDTH                    23:20
#define LWE397_TEX_TEXDESC_NPOT_AUX_0_LOG2_HEIGHT                   19:16
#define LWE397_TEX_TEXDESC_NPOT_AUX_0_LOD_MIN                       15:8
#define LWE397_TEX_TEXDESC_NPOT_AUX_0_LOD_MAX                       7:0


// Register LWE397_TEX_TEXDESC_NPOT_AUX_1  
#define LWE397_TEX_TEXDESC_NPOT_AUX_1                     (0x751)
#define LWE397_TEX_TEXDESC_NPOT_AUX_1_BASE_LEVEL_ONLY                       24:24
#define LWE397_TEX_TEXDESC_NPOT_AUX_1_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_NPOT_AUX_1_BASE_LEVEL_ONLY_ENABLE                      (1)
#define LWE397_TEX_TEXDESC_NPOT_AUX_1_LOG2_WIDTH                    23:20
#define LWE397_TEX_TEXDESC_NPOT_AUX_1_LOG2_HEIGHT                   19:16
#define LWE397_TEX_TEXDESC_NPOT_AUX_1_LOD_MIN                       15:8
#define LWE397_TEX_TEXDESC_NPOT_AUX_1_LOD_MAX                       7:0


// Register LWE397_TEX_TEXDESC_NPOT_AUX_2  
#define LWE397_TEX_TEXDESC_NPOT_AUX_2                     (0x752)
#define LWE397_TEX_TEXDESC_NPOT_AUX_2_BASE_LEVEL_ONLY                       24:24
#define LWE397_TEX_TEXDESC_NPOT_AUX_2_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_NPOT_AUX_2_BASE_LEVEL_ONLY_ENABLE                      (1)
#define LWE397_TEX_TEXDESC_NPOT_AUX_2_LOG2_WIDTH                    23:20
#define LWE397_TEX_TEXDESC_NPOT_AUX_2_LOG2_HEIGHT                   19:16
#define LWE397_TEX_TEXDESC_NPOT_AUX_2_LOD_MIN                       15:8
#define LWE397_TEX_TEXDESC_NPOT_AUX_2_LOD_MAX                       7:0


// Register LWE397_TEX_TEXDESC_NPOT_AUX_3  
#define LWE397_TEX_TEXDESC_NPOT_AUX_3                     (0x753)
#define LWE397_TEX_TEXDESC_NPOT_AUX_3_BASE_LEVEL_ONLY                       24:24
#define LWE397_TEX_TEXDESC_NPOT_AUX_3_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_NPOT_AUX_3_BASE_LEVEL_ONLY_ENABLE                      (1)
#define LWE397_TEX_TEXDESC_NPOT_AUX_3_LOG2_WIDTH                    23:20
#define LWE397_TEX_TEXDESC_NPOT_AUX_3_LOG2_HEIGHT                   19:16
#define LWE397_TEX_TEXDESC_NPOT_AUX_3_LOD_MIN                       15:8
#define LWE397_TEX_TEXDESC_NPOT_AUX_3_LOD_MAX                       7:0


// Register LWE397_TEX_TEXDESC_NPOT_AUX_4  
#define LWE397_TEX_TEXDESC_NPOT_AUX_4                     (0x754)
#define LWE397_TEX_TEXDESC_NPOT_AUX_4_BASE_LEVEL_ONLY                       24:24
#define LWE397_TEX_TEXDESC_NPOT_AUX_4_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_NPOT_AUX_4_BASE_LEVEL_ONLY_ENABLE                      (1)
#define LWE397_TEX_TEXDESC_NPOT_AUX_4_LOG2_WIDTH                    23:20

#define LWE397_TEX_TEXDESC_NPOT_AUX_4_LOG2_HEIGHT                   19:16
#define LWE397_TEX_TEXDESC_NPOT_AUX_4_LOD_MIN                       15:8
#define LWE397_TEX_TEXDESC_NPOT_AUX_4_LOD_MAX                       7:0


// Register LWE397_TEX_TEXDESC_NPOT_AUX_5  
#define LWE397_TEX_TEXDESC_NPOT_AUX_5                     (0x755)
#define LWE397_TEX_TEXDESC_NPOT_AUX_5_BASE_LEVEL_ONLY                       24:24
#define LWE397_TEX_TEXDESC_NPOT_AUX_5_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_NPOT_AUX_5_BASE_LEVEL_ONLY_ENABLE                      (1)
#define LWE397_TEX_TEXDESC_NPOT_AUX_5_LOG2_WIDTH                    23:20
#define LWE397_TEX_TEXDESC_NPOT_AUX_5_LOG2_HEIGHT                   19:16
#define LWE397_TEX_TEXDESC_NPOT_AUX_5_LOD_MIN                       15:8
#define LWE397_TEX_TEXDESC_NPOT_AUX_5_LOD_MAX                       7:0


// Register LWE397_TEX_TEXDESC_NPOT_AUX_6  
#define LWE397_TEX_TEXDESC_NPOT_AUX_6                     (0x756)
#define LWE397_TEX_TEXDESC_NPOT_AUX_6_BASE_LEVEL_ONLY                       24:24
#define LWE397_TEX_TEXDESC_NPOT_AUX_6_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_NPOT_AUX_6_BASE_LEVEL_ONLY_ENABLE                      (1)
#define LWE397_TEX_TEXDESC_NPOT_AUX_6_LOG2_WIDTH                    23:20
#define LWE397_TEX_TEXDESC_NPOT_AUX_6_LOG2_HEIGHT                   19:16
#define LWE397_TEX_TEXDESC_NPOT_AUX_6_LOD_MIN                       15:8
#define LWE397_TEX_TEXDESC_NPOT_AUX_6_LOD_MAX                       7:0


// Register LWE397_TEX_TEXDESC_NPOT_AUX_7  
#define LWE397_TEX_TEXDESC_NPOT_AUX_7                     (0x757)
#define LWE397_TEX_TEXDESC_NPOT_AUX_7_BASE_LEVEL_ONLY                       24:24
#define LWE397_TEX_TEXDESC_NPOT_AUX_7_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_NPOT_AUX_7_BASE_LEVEL_ONLY_ENABLE                      (1)
#define LWE397_TEX_TEXDESC_NPOT_AUX_7_LOG2_WIDTH                    23:20
#define LWE397_TEX_TEXDESC_NPOT_AUX_7_LOG2_HEIGHT                   19:16
#define LWE397_TEX_TEXDESC_NPOT_AUX_7_LOD_MIN                       15:8
#define LWE397_TEX_TEXDESC_NPOT_AUX_7_LOD_MAX                       7:0


// Register LWE397_TEX_TEXDESC_NPOT_AUX_8  
#define LWE397_TEX_TEXDESC_NPOT_AUX_8                     (0x758)
#define LWE397_TEX_TEXDESC_NPOT_AUX_8_BASE_LEVEL_ONLY                       24:24
#define LWE397_TEX_TEXDESC_NPOT_AUX_8_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_NPOT_AUX_8_BASE_LEVEL_ONLY_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_NPOT_AUX_8_LOG2_WIDTH                    23:20

#define LWE397_TEX_TEXDESC_NPOT_AUX_8_LOG2_HEIGHT                   19:16

#define LWE397_TEX_TEXDESC_NPOT_AUX_8_LOD_MIN                       15:8

#define LWE397_TEX_TEXDESC_NPOT_AUX_8_LOD_MAX                       7:0


// Register LWE397_TEX_TEXDESC_NPOT_AUX_9  
#define LWE397_TEX_TEXDESC_NPOT_AUX_9                     (0x759)
#define LWE397_TEX_TEXDESC_NPOT_AUX_9_BASE_LEVEL_ONLY                       24:24
#define LWE397_TEX_TEXDESC_NPOT_AUX_9_BASE_LEVEL_ONLY_DISABLE                     (0)
#define LWE397_TEX_TEXDESC_NPOT_AUX_9_BASE_LEVEL_ONLY_ENABLE                      (1)

#define LWE397_TEX_TEXDESC_NPOT_AUX_9_LOG2_WIDTH                    23:20

#define LWE397_TEX_TEXDESC_NPOT_AUX_9_LOG2_HEIGHT                   19:16

#define LWE397_TEX_TEXDESC_NPOT_AUX_9_LOD_MIN                       15:8

#define LWE397_TEX_TEXDESC_NPOT_AUX_9_LOD_MAX                       7:0


// Register LWE397_TEX_TEXDESC_NPOT_AUX_10  
#define LWE397_TEX_TEXDESC_NPOT_AUX_10                    (0x75a)
#define LWE397_TEX_TEXDESC_NPOT_AUX_10_BASE_LEVEL_ONLY                      24:24
#define LWE397_TEX_TEXDESC_NPOT_AUX_10_BASE_LEVEL_ONLY_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_NPOT_AUX_10_BASE_LEVEL_ONLY_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_NPOT_AUX_10_LOG2_WIDTH                   23:20

#define LWE397_TEX_TEXDESC_NPOT_AUX_10_LOG2_HEIGHT                  19:16

#define LWE397_TEX_TEXDESC_NPOT_AUX_10_LOD_MIN                      15:8

#define LWE397_TEX_TEXDESC_NPOT_AUX_10_LOD_MAX                      7:0


// Register LWE397_TEX_TEXDESC_NPOT_AUX_11  
#define LWE397_TEX_TEXDESC_NPOT_AUX_11                    (0x75b)
#define LWE397_TEX_TEXDESC_NPOT_AUX_11_BASE_LEVEL_ONLY                      24:24
#define LWE397_TEX_TEXDESC_NPOT_AUX_11_BASE_LEVEL_ONLY_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_NPOT_AUX_11_BASE_LEVEL_ONLY_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_NPOT_AUX_11_LOG2_WIDTH                   23:20

#define LWE397_TEX_TEXDESC_NPOT_AUX_11_LOG2_HEIGHT                  19:16

#define LWE397_TEX_TEXDESC_NPOT_AUX_11_LOD_MIN                      15:8

#define LWE397_TEX_TEXDESC_NPOT_AUX_11_LOD_MAX                      7:0


// Register LWE397_TEX_TEXDESC_NPOT_AUX_12  
#define LWE397_TEX_TEXDESC_NPOT_AUX_12                    (0x75c)
#define LWE397_TEX_TEXDESC_NPOT_AUX_12_BASE_LEVEL_ONLY                      24:24
#define LWE397_TEX_TEXDESC_NPOT_AUX_12_BASE_LEVEL_ONLY_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_NPOT_AUX_12_BASE_LEVEL_ONLY_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_NPOT_AUX_12_LOG2_WIDTH                   23:20

#define LWE397_TEX_TEXDESC_NPOT_AUX_12_LOG2_HEIGHT                  19:16

#define LWE397_TEX_TEXDESC_NPOT_AUX_12_LOD_MIN                      15:8

#define LWE397_TEX_TEXDESC_NPOT_AUX_12_LOD_MAX                      7:0


// Register LWE397_TEX_TEXDESC_NPOT_AUX_13  
#define LWE397_TEX_TEXDESC_NPOT_AUX_13                    (0x75d)
#define LWE397_TEX_TEXDESC_NPOT_AUX_13_BASE_LEVEL_ONLY                      24:24
#define LWE397_TEX_TEXDESC_NPOT_AUX_13_BASE_LEVEL_ONLY_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_NPOT_AUX_13_BASE_LEVEL_ONLY_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_NPOT_AUX_13_LOG2_WIDTH                   23:20

#define LWE397_TEX_TEXDESC_NPOT_AUX_13_LOG2_HEIGHT                  19:16

#define LWE397_TEX_TEXDESC_NPOT_AUX_13_LOD_MIN                      15:8

#define LWE397_TEX_TEXDESC_NPOT_AUX_13_LOD_MAX                      7:0


// Register LWE397_TEX_TEXDESC_NPOT_AUX_14  
#define LWE397_TEX_TEXDESC_NPOT_AUX_14                    (0x75e)
#define LWE397_TEX_TEXDESC_NPOT_AUX_14_BASE_LEVEL_ONLY                      24:24
#define LWE397_TEX_TEXDESC_NPOT_AUX_14_BASE_LEVEL_ONLY_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_NPOT_AUX_14_BASE_LEVEL_ONLY_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_NPOT_AUX_14_LOG2_WIDTH                   23:20

#define LWE397_TEX_TEXDESC_NPOT_AUX_14_LOG2_HEIGHT                  19:16

#define LWE397_TEX_TEXDESC_NPOT_AUX_14_LOD_MIN                      15:8

#define LWE397_TEX_TEXDESC_NPOT_AUX_14_LOD_MAX                      7:0


// Register LWE397_TEX_TEXDESC_NPOT_AUX_15  
#define LWE397_TEX_TEXDESC_NPOT_AUX_15                    (0x75f)
#define LWE397_TEX_TEXDESC_NPOT_AUX_15_BASE_LEVEL_ONLY                      24:24
#define LWE397_TEX_TEXDESC_NPOT_AUX_15_BASE_LEVEL_ONLY_DISABLE                    (0)
#define LWE397_TEX_TEXDESC_NPOT_AUX_15_BASE_LEVEL_ONLY_ENABLE                     (1)

#define LWE397_TEX_TEXDESC_NPOT_AUX_15_LOG2_WIDTH                   23:20

#define LWE397_TEX_TEXDESC_NPOT_AUX_15_LOG2_HEIGHT                  19:16

#define LWE397_TEX_TEXDESC_NPOT_AUX_15_LOD_MIN                      15:8

#define LWE397_TEX_TEXDESC_NPOT_AUX_15_LOD_MAX                      7:0


// Register LWE397_ALU_REMAP_OFFSET_0  
#define LWE397_ALU_REMAP_OFFSET_0                 (0x800)
#define LWE397_ALU_REMAP_OFFSET_0_INDEX                     5:0

#define LWE397_ALU_REMAP_OFFSET_0_BASE                      11:6


// Register LWE397_ALU_REMAP_DATA_0  
#define LWE397_ALU_REMAP_DATA_0                   (0x801)
#define LWE397_ALU_REMAP_DATA_0_COUNT                       1:0

#define LWE397_ALU_REMAP_DATA_0_OFFSET                      7:2


// Register LWE397_ALU_REMAP_DATA_4X_0  
#define LWE397_ALU_REMAP_DATA_4X_0                        (0x802)
#define LWE397_ALU_REMAP_DATA_4X_0_COUNT0                   1:0

#define LWE397_ALU_REMAP_DATA_4X_0_OFFSET0                  7:2

#define LWE397_ALU_REMAP_DATA_4X_0_COUNT1                   9:8

#define LWE397_ALU_REMAP_DATA_4X_0_OFFSET1                  15:10

#define LWE397_ALU_REMAP_DATA_4X_0_COUNT2                   17:16

#define LWE397_ALU_REMAP_DATA_4X_0_OFFSET2                  23:18

#define LWE397_ALU_REMAP_DATA_4X_0_COUNT3                   25:24

#define LWE397_ALU_REMAP_DATA_4X_0_OFFSET3                  31:26


// Register LWE397_ALU_INST_OFFSET_0  
#define LWE397_ALU_INST_OFFSET_0                  (0x803)
#define LWE397_ALU_INST_OFFSET_0_INDEX                      8:0


// Register LWE397_ALU_INST_DATA_0  
#define LWE397_ALU_INST_DATA_0                    (0x804)
#define LWE397_ALU_INST_DATA_0_OP                   31:27

#define LWE397_ALU_INST_DATA_0_DEST                 26:13

#define LWE397_ALU_INST_DATA_0_A                    12:0

#define LWE397_ALU_INST_DATA_0_B                    31:19

#define LWE397_ALU_INST_DATA_0_C                    18:6

#define LWE397_ALU_INST_DATA_0_D                    5:0


// Packet LWE397_ALU_ARGSEL_ABC
#define LWE397_ALU_ARGSEL_ABC_SIZE 13

#define LWE397_ALU_ARGSEL_ABC_RSEL_ROW                    0
#define LWE397_ALU_ARGSEL_ABC_RSEL_R0                     (0)
#define LWE397_ALU_ARGSEL_ABC_RSEL_R1                     (1)
#define LWE397_ALU_ARGSEL_ABC_RSEL_R2                     (2)
#define LWE397_ALU_ARGSEL_ABC_RSEL_R3                     (3)
#define LWE397_ALU_ARGSEL_ABC_RSEL_R4                     (4)
#define LWE397_ALU_ARGSEL_ABC_RSEL_R5                     (5)
#define LWE397_ALU_ARGSEL_ABC_RSEL_R6                     (6)
#define LWE397_ALU_ARGSEL_ABC_RSEL_R7                     (7)
#define LWE397_ALU_ARGSEL_ABC_RSEL_R8                     (8)
#define LWE397_ALU_ARGSEL_ABC_RSEL_R9                     (9)
#define LWE397_ALU_ARGSEL_ABC_RSEL_R10                    (10)
#define LWE397_ALU_ARGSEL_ABC_RSEL_R11                    (11)
#define LWE397_ALU_ARGSEL_ABC_RSEL_R12                    (12)
#define LWE397_ALU_ARGSEL_ABC_RSEL_R13                    (13)
#define LWE397_ALU_ARGSEL_ABC_RSEL_R14                    (14)
#define LWE397_ALU_ARGSEL_ABC_RSEL_R15                    (15)
#define LWE397_ALU_ARGSEL_ABC_RSEL_S0                     (16)
#define LWE397_ALU_ARGSEL_ABC_RSEL_S1                     (17)
#define LWE397_ALU_ARGSEL_ABC_RSEL_S2                     (18)
#define LWE397_ALU_ARGSEL_ABC_RSEL_S3                     (19)
#define LWE397_ALU_ARGSEL_ABC_RSEL_S4                     (20)
#define LWE397_ALU_ARGSEL_ABC_RSEL_S5                     (21)
#define LWE397_ALU_ARGSEL_ABC_RSEL_S6                     (22)
#define LWE397_ALU_ARGSEL_ABC_RSEL_S7                     (23)
#define LWE397_ALU_ARGSEL_ABC_RSEL_RES0                   (24)
#define LWE397_ALU_ARGSEL_ABC_RSEL_RES1                   (25)
#define LWE397_ALU_ARGSEL_ABC_RSEL_RES2                   (26)
#define LWE397_ALU_ARGSEL_ABC_RSEL_RES3                   (27)
#define LWE397_ALU_ARGSEL_ABC_RSEL_IMM0                   (28)
#define LWE397_ALU_ARGSEL_ABC_RSEL_IMM1                   (29)
#define LWE397_ALU_ARGSEL_ABC_RSEL_IMM2                   (30)
#define LWE397_ALU_ARGSEL_ABC_RSEL_C01                    (31)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G0                     (32)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G1                     (33)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G2                     (34)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G3                     (35)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G4                     (36)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G5                     (37)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G6                     (38)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G7                     (39)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G8                     (40)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G9                     (41)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G10                    (42)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G11                    (43)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G12                    (44)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G13                    (45)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G14                    (46)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G15                    (47)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G16                    (48)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G17                    (49)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G18                    (50)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G19                    (51)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G20                    (52)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G21                    (53)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G22                    (54)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G23                    (55)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G24                    (56)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G25                    (57)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G26                    (58)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G27                    (59)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G28                    (60)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G29                    (61)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G30                    (62)
#define LWE397_ALU_ARGSEL_ABC_RSEL_G31                    (63)
#define LWE397_ALU_ARGSEL_ABC_RSEL_P0                     (64)
#define LWE397_ALU_ARGSEL_ABC_RSEL_P1                     (65)
#define LWE397_ALU_ARGSEL_ABC_RSEL_P2                     (66)
#define LWE397_ALU_ARGSEL_ABC_RSEL_P3                     (67)
#define LWE397_ALU_ARGSEL_ABC_RSEL_P4                     (68)
#define LWE397_ALU_ARGSEL_ABC_RSEL_P5                     (69)
#define LWE397_ALU_ARGSEL_ABC_RSEL_P6                     (70)
#define LWE397_ALU_ARGSEL_ABC_RSEL_P7                     (71)
#define LWE397_ALU_ARGSEL_ABC_RSEL_X                      (72)
#define LWE397_ALU_ARGSEL_ABC_RSEL_Y                      (73)
#define LWE397_ALU_ARGSEL_ABC_RSEL_PCOV                   (74)
#define LWE397_ALU_ARGSEL_ABC_RSEL_FF                     (75)
#define LWE397_ALU_ARGSEL_ABC_RSEL_KC                     (76)

#define LWE397_ALU_ARGSEL_ABC_MOD_ROW                     0
#define LWE397_ALU_ARGSEL_ABC_MOD_L                       (0)
#define LWE397_ALU_ARGSEL_ABC_MOD_H                       (1)

#define LWE397_ALU_ARGSEL_ABC_SUB_1_ROW                   0

#define LWE397_ALU_ARGSEL_ABC_PREC_ROW                    0
#define LWE397_ALU_ARGSEL_ABC_PREC_FP                     (0)
#define LWE397_ALU_ARGSEL_ABC_PREC_FX                     (1)

#define LWE397_ALU_ARGSEL_ABC_ABS_ROW                     0

#define LWE397_ALU_ARGSEL_ABC_NEG_ROW                     0

#define LWE397_ALU_ARGSEL_ABC_X2_ROW                      0


// Packet LWE397_ALU_ARGSEL_D
#define LWE397_ALU_ARGSEL_D_SIZE 6

#define LWE397_ALU_ARGSEL_D_RSEL_ROW                      0
#define LWE397_ALU_ARGSEL_D_RSEL_B                        (0)
#define LWE397_ALU_ARGSEL_D_RSEL_C                        (1)

#define LWE397_ALU_ARGSEL_D_MOD_ROW                       0
#define LWE397_ALU_ARGSEL_D_MOD_L                 (0)
#define LWE397_ALU_ARGSEL_D_MOD_H                 (1)

#define LWE397_ALU_ARGSEL_D_SUB_1_ROW                     0

#define LWE397_ALU_ARGSEL_D_FUNC_ROW                      0
#define LWE397_ALU_ARGSEL_D_FUNC_MAD                      (0)
#define LWE397_ALU_ARGSEL_D_FUNC_DP2                      (1)

#define LWE397_ALU_ARGSEL_D_ABS_ROW                       0

#define LWE397_ALU_ARGSEL_D_PREC_ROW                      0
#define LWE397_ALU_ARGSEL_D_PREC_FP                       (0)
#define LWE397_ALU_ARGSEL_D_PREC_FX                       (1)

#define ALU_XY_EXP      44

// Packet LWE397_ALU_COORD
#define LWE397_ALU_COORD_SIZE 20

#define LWE397_ALU_COORD_EXP_ROW                  0

#define LWE397_ALU_COORD_COORD_ROW                        0

#define LWE397_ALU_COORD_FP20_ROW                 0


// Packet LWE397_ALU_OP
#define LWE397_ALU_OP_SIZE 5

#define LWE397_ALU_OP_OP_ROW                      0
#define LWE397_ALU_OP_OP_MAD                      (0)
#define LWE397_ALU_OP_OP_MIN                      (1)
#define LWE397_ALU_OP_OP_MAX                      (2)
#define LWE397_ALU_OP_OP_CMP                      (3)

#define LWE397_ALU_OP_SEND_ROW                    0

#define LWE397_ALU_OP_RECV_ROW                    0

#define LWE397_ALU_OP_MMUL_ROW                    0


// Packet LWE397_ALU_RESSEL
#define LWE397_ALU_RESSEL_SIZE 14

#define LWE397_ALU_RESSEL_SCALE_ROW                       0
#define LWE397_ALU_RESSEL_SCALE_X1                        (0)
#define LWE397_ALU_RESSEL_SCALE_X2                        (1)
#define LWE397_ALU_RESSEL_SCALE_X4                        (2)
#define LWE397_ALU_RESSEL_SCALE_DIV2                      (3)

#define LWE397_ALU_RESSEL_CLAMP_ROW                       0

#define LWE397_ALU_RESSEL_CC_ROW                  0
#define LWE397_ALU_RESSEL_CC_NONE                 (0)
#define LWE397_ALU_RESSEL_CC_SEQ                  (1)
#define LWE397_ALU_RESSEL_CC_SGT                  (2)
#define LWE397_ALU_RESSEL_CC_SGE                  (3)

#define LWE397_ALU_RESSEL_RSEL_ROW                        0
#define LWE397_ALU_RESSEL_RSEL_R0                 (0)
#define LWE397_ALU_RESSEL_RSEL_R1                 (1)
#define LWE397_ALU_RESSEL_RSEL_R2                 (2)
#define LWE397_ALU_RESSEL_RSEL_R3                 (3)
#define LWE397_ALU_RESSEL_RSEL_R4                 (4)
#define LWE397_ALU_RESSEL_RSEL_R5                 (5)
#define LWE397_ALU_RESSEL_RSEL_R6                 (6)
#define LWE397_ALU_RESSEL_RSEL_R7                 (7)
#define LWE397_ALU_RESSEL_RSEL_R8                 (8)
#define LWE397_ALU_RESSEL_RSEL_R9                 (9)
#define LWE397_ALU_RESSEL_RSEL_R10                        (10)
#define LWE397_ALU_RESSEL_RSEL_R11                        (11)
#define LWE397_ALU_RESSEL_RSEL_R12                        (12)
#define LWE397_ALU_RESSEL_RSEL_R13                        (13)
#define LWE397_ALU_RESSEL_RSEL_R14                        (14)
#define LWE397_ALU_RESSEL_RSEL_R15                        (15)
#define LWE397_ALU_RESSEL_RSEL_S0                 (16)
#define LWE397_ALU_RESSEL_RSEL_S1                 (17)
#define LWE397_ALU_RESSEL_RSEL_S2                 (18)
#define LWE397_ALU_RESSEL_RSEL_S3                 (19)
#define LWE397_ALU_RESSEL_RSEL_S4                 (20)
#define LWE397_ALU_RESSEL_RSEL_S5                 (21)
#define LWE397_ALU_RESSEL_RSEL_S6                 (22)
#define LWE397_ALU_RESSEL_RSEL_S7                 (23)
#define LWE397_ALU_RESSEL_RSEL_RES0                       (24)
#define LWE397_ALU_RESSEL_RSEL_RES1                       (25)
#define LWE397_ALU_RESSEL_RSEL_RES2                       (26)
#define LWE397_ALU_RESSEL_RSEL_RES3                       (27)
#define LWE397_ALU_RESSEL_RSEL_IMM0                       (28)
#define LWE397_ALU_RESSEL_RSEL_IMM1                       (29)
#define LWE397_ALU_RESSEL_RSEL_IMM2                       (30)
#define LWE397_ALU_RESSEL_RSEL_C01                        (31)
#define LWE397_ALU_RESSEL_RSEL_G0                 (32)
#define LWE397_ALU_RESSEL_RSEL_G1                 (33)
#define LWE397_ALU_RESSEL_RSEL_G2                 (34)
#define LWE397_ALU_RESSEL_RSEL_G3                 (35)
#define LWE397_ALU_RESSEL_RSEL_G4                 (36)
#define LWE397_ALU_RESSEL_RSEL_G5                 (37)
#define LWE397_ALU_RESSEL_RSEL_G6                 (38)
#define LWE397_ALU_RESSEL_RSEL_G7                 (39)
#define LWE397_ALU_RESSEL_RSEL_G8                 (40)
#define LWE397_ALU_RESSEL_RSEL_G9                 (41)
#define LWE397_ALU_RESSEL_RSEL_G10                        (42)
#define LWE397_ALU_RESSEL_RSEL_G11                        (43)
#define LWE397_ALU_RESSEL_RSEL_G12                        (44)
#define LWE397_ALU_RESSEL_RSEL_G13                        (45)
#define LWE397_ALU_RESSEL_RSEL_G14                        (46)
#define LWE397_ALU_RESSEL_RSEL_G15                        (47)
#define LWE397_ALU_RESSEL_RSEL_G16                        (48)
#define LWE397_ALU_RESSEL_RSEL_G17                        (49)
#define LWE397_ALU_RESSEL_RSEL_G18                        (50)
#define LWE397_ALU_RESSEL_RSEL_G19                        (51)
#define LWE397_ALU_RESSEL_RSEL_G20                        (52)
#define LWE397_ALU_RESSEL_RSEL_G21                        (53)
#define LWE397_ALU_RESSEL_RSEL_G22                        (54)
#define LWE397_ALU_RESSEL_RSEL_G23                        (55)
#define LWE397_ALU_RESSEL_RSEL_G24                        (56)
#define LWE397_ALU_RESSEL_RSEL_G25                        (57)
#define LWE397_ALU_RESSEL_RSEL_G26                        (58)
#define LWE397_ALU_RESSEL_RSEL_G27                        (59)
#define LWE397_ALU_RESSEL_RSEL_G28                        (60)
#define LWE397_ALU_RESSEL_RSEL_G29                        (61)
#define LWE397_ALU_RESSEL_RSEL_G30                        (62)
#define LWE397_ALU_RESSEL_RSEL_G31                        (63)
#define LWE397_ALU_RESSEL_RSEL_P0                 (64)
#define LWE397_ALU_RESSEL_RSEL_P1                 (65)
#define LWE397_ALU_RESSEL_RSEL_P2                 (66)
#define LWE397_ALU_RESSEL_RSEL_P3                 (67)
#define LWE397_ALU_RESSEL_RSEL_P4                 (68)
#define LWE397_ALU_RESSEL_RSEL_P5                 (69)
#define LWE397_ALU_RESSEL_RSEL_P6                 (70)
#define LWE397_ALU_RESSEL_RSEL_P7                 (71)
#define LWE397_ALU_RESSEL_RSEL_X                  (72)
#define LWE397_ALU_RESSEL_RSEL_Y                  (73)
#define LWE397_ALU_RESSEL_RSEL_PCOV                       (74)
#define LWE397_ALU_RESSEL_RSEL_FF                 (75)
#define LWE397_ALU_RESSEL_RSEL_KC                 (76)

#define LWE397_ALU_RESSEL_DSTSEL_ROW                      0
#define LWE397_ALU_RESSEL_DSTSEL_FP20                     (3)
#define LWE397_ALU_RESSEL_DSTSEL_FX10_H                   (2)
#define LWE397_ALU_RESSEL_DSTSEL_FX10_L                   (1)
#define LWE397_ALU_RESSEL_DSTSEL_FP20_L                   (0)


// Register LWE397_ALU_P2CX_OFFSET_0  
#define LWE397_ALU_P2CX_OFFSET_0                  (0x805)
#define LWE397_ALU_P2CX_OFFSET_0_INDEX                      5:0


// Register LWE397_ALU_P2CX_DATA_0  
#define LWE397_ALU_P2CX_DATA_0                    (0x806)
#define LWE397_ALU_P2CX_DATA_0_MASK                 31:16

#define LWE397_ALU_P2CX_DATA_0_REFERENCE                    15:0

#define LWE397_ALU_P2CX_DATA_0_OPCODE                       31:0
#define LWE397_ALU_P2CX_DATA_0_OPCODE_KEEP                        (0)
#define LWE397_ALU_P2CX_DATA_0_OPCODE_ILWERT                      (1)
#define LWE397_ALU_P2CX_DATA_0_OPCODE_EXELWTE                     (2)
#define LWE397_ALU_P2CX_DATA_0_OPCODE_NO_EXELWTE                  (3)


// Register LWE397_ALU_GLOBALS_0  
#define LWE397_ALU_GLOBALS_0                      (0x820)
#define LWE397_ALU_GLOBALS_0_L10                    9:0

#define LWE397_ALU_GLOBALS_0_H10                    19:10

#define LWE397_ALU_GLOBALS_0_VAL                    19:0


// Register LWE397_ALU_GLOBALS  
#define LWE397_ALU_GLOBALS                        (0x820)
#define LWE397_ALU_GLOBALS_L10                      9:0

#define LWE397_ALU_GLOBALS_H10                      19:10

#define LWE397_ALU_GLOBALS_VAL                      19:0


// Register LWE397_ALU_GLOBALS_1  
#define LWE397_ALU_GLOBALS_1                      (0x821)
#define LWE397_ALU_GLOBALS_1_L10                    9:0

#define LWE397_ALU_GLOBALS_1_H10                    19:10

#define LWE397_ALU_GLOBALS_1_VAL                    19:0


// Register LWE397_ALU_GLOBALS_2  
#define LWE397_ALU_GLOBALS_2                      (0x822)
#define LWE397_ALU_GLOBALS_2_L10                    9:0

#define LWE397_ALU_GLOBALS_2_H10                    19:10

#define LWE397_ALU_GLOBALS_2_VAL                    19:0


// Register LWE397_ALU_GLOBALS_3  
#define LWE397_ALU_GLOBALS_3                      (0x823)
#define LWE397_ALU_GLOBALS_3_L10                    9:0

#define LWE397_ALU_GLOBALS_3_H10                    19:10

#define LWE397_ALU_GLOBALS_3_VAL                    19:0


// Register LWE397_ALU_GLOBALS_4  
#define LWE397_ALU_GLOBALS_4                      (0x824)
#define LWE397_ALU_GLOBALS_4_L10                    9:0

#define LWE397_ALU_GLOBALS_4_H10                    19:10

#define LWE397_ALU_GLOBALS_4_VAL                    19:0


// Register LWE397_ALU_GLOBALS_5  
#define LWE397_ALU_GLOBALS_5                      (0x825)
#define LWE397_ALU_GLOBALS_5_L10                    9:0

#define LWE397_ALU_GLOBALS_5_H10                    19:10

#define LWE397_ALU_GLOBALS_5_VAL                    19:0


// Register LWE397_ALU_GLOBALS_6  
#define LWE397_ALU_GLOBALS_6                      (0x826)
#define LWE397_ALU_GLOBALS_6_L10                    9:0

#define LWE397_ALU_GLOBALS_6_H10                    19:10

#define LWE397_ALU_GLOBALS_6_VAL                    19:0


// Register LWE397_ALU_GLOBALS_7  
#define LWE397_ALU_GLOBALS_7                      (0x827)
#define LWE397_ALU_GLOBALS_7_L10                    9:0

#define LWE397_ALU_GLOBALS_7_H10                    19:10

#define LWE397_ALU_GLOBALS_7_VAL                    19:0


// Register LWE397_ALU_GLOBALS_8  
#define LWE397_ALU_GLOBALS_8                      (0x828)
#define LWE397_ALU_GLOBALS_8_L10                    9:0

#define LWE397_ALU_GLOBALS_8_H10                    19:10

#define LWE397_ALU_GLOBALS_8_VAL                    19:0


// Register LWE397_ALU_GLOBALS_9  
#define LWE397_ALU_GLOBALS_9                      (0x829)
#define LWE397_ALU_GLOBALS_9_L10                    9:0

#define LWE397_ALU_GLOBALS_9_H10                    19:10

#define LWE397_ALU_GLOBALS_9_VAL                    19:0


// Register LWE397_ALU_GLOBALS_10  
#define LWE397_ALU_GLOBALS_10                     (0x82a)
#define LWE397_ALU_GLOBALS_10_L10                   9:0

#define LWE397_ALU_GLOBALS_10_H10                   19:10

#define LWE397_ALU_GLOBALS_10_VAL                   19:0


// Register LWE397_ALU_GLOBALS_11  
#define LWE397_ALU_GLOBALS_11                     (0x82b)
#define LWE397_ALU_GLOBALS_11_L10                   9:0

#define LWE397_ALU_GLOBALS_11_H10                   19:10

#define LWE397_ALU_GLOBALS_11_VAL                   19:0


// Register LWE397_ALU_GLOBALS_12  
#define LWE397_ALU_GLOBALS_12                     (0x82c)
#define LWE397_ALU_GLOBALS_12_L10                   9:0

#define LWE397_ALU_GLOBALS_12_H10                   19:10

#define LWE397_ALU_GLOBALS_12_VAL                   19:0


// Register LWE397_ALU_GLOBALS_13  
#define LWE397_ALU_GLOBALS_13                     (0x82d)
#define LWE397_ALU_GLOBALS_13_L10                   9:0

#define LWE397_ALU_GLOBALS_13_H10                   19:10

#define LWE397_ALU_GLOBALS_13_VAL                   19:0


// Register LWE397_ALU_GLOBALS_14  
#define LWE397_ALU_GLOBALS_14                     (0x82e)
#define LWE397_ALU_GLOBALS_14_L10                   9:0

#define LWE397_ALU_GLOBALS_14_H10                   19:10

#define LWE397_ALU_GLOBALS_14_VAL                   19:0


// Register LWE397_ALU_GLOBALS_15  
#define LWE397_ALU_GLOBALS_15                     (0x82f)
#define LWE397_ALU_GLOBALS_15_L10                   9:0

#define LWE397_ALU_GLOBALS_15_H10                   19:10

#define LWE397_ALU_GLOBALS_15_VAL                   19:0


// Register LWE397_ALU_GLOBALS_16  
#define LWE397_ALU_GLOBALS_16                     (0x830)
#define LWE397_ALU_GLOBALS_16_L10                   9:0

#define LWE397_ALU_GLOBALS_16_H10                   19:10

#define LWE397_ALU_GLOBALS_16_VAL                   19:0


// Register LWE397_ALU_GLOBALS_17  
#define LWE397_ALU_GLOBALS_17                     (0x831)
#define LWE397_ALU_GLOBALS_17_L10                   9:0

#define LWE397_ALU_GLOBALS_17_H10                   19:10

#define LWE397_ALU_GLOBALS_17_VAL                   19:0


// Register LWE397_ALU_GLOBALS_18  
#define LWE397_ALU_GLOBALS_18                     (0x832)
#define LWE397_ALU_GLOBALS_18_L10                   9:0

#define LWE397_ALU_GLOBALS_18_H10                   19:10

#define LWE397_ALU_GLOBALS_18_VAL                   19:0


// Register LWE397_ALU_GLOBALS_19  
#define LWE397_ALU_GLOBALS_19                     (0x833)
#define LWE397_ALU_GLOBALS_19_L10                   9:0

#define LWE397_ALU_GLOBALS_19_H10                   19:10

#define LWE397_ALU_GLOBALS_19_VAL                   19:0


// Register LWE397_ALU_GLOBALS_20  
#define LWE397_ALU_GLOBALS_20                     (0x834)
#define LWE397_ALU_GLOBALS_20_L10                   9:0

#define LWE397_ALU_GLOBALS_20_H10                   19:10

#define LWE397_ALU_GLOBALS_20_VAL                   19:0


// Register LWE397_ALU_GLOBALS_21  
#define LWE397_ALU_GLOBALS_21                     (0x835)
#define LWE397_ALU_GLOBALS_21_L10                   9:0

#define LWE397_ALU_GLOBALS_21_H10                   19:10

#define LWE397_ALU_GLOBALS_21_VAL                   19:0


// Register LWE397_ALU_GLOBALS_22  
#define LWE397_ALU_GLOBALS_22                     (0x836)
#define LWE397_ALU_GLOBALS_22_L10                   9:0

#define LWE397_ALU_GLOBALS_22_H10                   19:10

#define LWE397_ALU_GLOBALS_22_VAL                   19:0


// Register LWE397_ALU_GLOBALS_23  
#define LWE397_ALU_GLOBALS_23                     (0x837)
#define LWE397_ALU_GLOBALS_23_L10                   9:0

#define LWE397_ALU_GLOBALS_23_H10                   19:10

#define LWE397_ALU_GLOBALS_23_VAL                   19:0


// Register LWE397_ALU_GLOBALS_24  
#define LWE397_ALU_GLOBALS_24                     (0x838)
#define LWE397_ALU_GLOBALS_24_L10                   9:0

#define LWE397_ALU_GLOBALS_24_H10                   19:10

#define LWE397_ALU_GLOBALS_24_VAL                   19:0


// Register LWE397_ALU_GLOBALS_25  
#define LWE397_ALU_GLOBALS_25                     (0x839)
#define LWE397_ALU_GLOBALS_25_L10                   9:0

#define LWE397_ALU_GLOBALS_25_H10                   19:10

#define LWE397_ALU_GLOBALS_25_VAL                   19:0


// Register LWE397_ALU_GLOBALS_26  
#define LWE397_ALU_GLOBALS_26                     (0x83a)
#define LWE397_ALU_GLOBALS_26_L10                   9:0

#define LWE397_ALU_GLOBALS_26_H10                   19:10

#define LWE397_ALU_GLOBALS_26_VAL                   19:0


// Register LWE397_ALU_GLOBALS_27  
#define LWE397_ALU_GLOBALS_27                     (0x83b)
#define LWE397_ALU_GLOBALS_27_L10                   9:0

#define LWE397_ALU_GLOBALS_27_H10                   19:10

#define LWE397_ALU_GLOBALS_27_VAL                   19:0


// Register LWE397_ALU_GLOBALS_28  
#define LWE397_ALU_GLOBALS_28                     (0x83c)
#define LWE397_ALU_GLOBALS_28_L10                   9:0

#define LWE397_ALU_GLOBALS_28_H10                   19:10

#define LWE397_ALU_GLOBALS_28_VAL                   19:0


// Register LWE397_ALU_GLOBALS_29  
#define LWE397_ALU_GLOBALS_29                     (0x83d)
#define LWE397_ALU_GLOBALS_29_L10                   9:0

#define LWE397_ALU_GLOBALS_29_H10                   19:10

#define LWE397_ALU_GLOBALS_29_VAL                   19:0


// Register LWE397_ALU_GLOBALS_30  
#define LWE397_ALU_GLOBALS_30                     (0x83e)
#define LWE397_ALU_GLOBALS_30_L10                   9:0

#define LWE397_ALU_GLOBALS_30_H10                   19:10

#define LWE397_ALU_GLOBALS_30_VAL                   19:0


// Register LWE397_ALU_GLOBALS_31  
#define LWE397_ALU_GLOBALS_31                     (0x83f)
#define LWE397_ALU_GLOBALS_31_L10                   9:0

#define LWE397_ALU_GLOBALS_31_H10                   19:10

#define LWE397_ALU_GLOBALS_31_VAL                   19:0


// Packet LWE397_ALU_IMMEDIATE
#define LWE397_ALU_IMMEDIATE_SIZE 60

#define LWE397_ALU_IMMEDIATE_IMM2FP_ROW                   0

#define LWE397_ALU_IMMEDIATE_IMM2FXH_ROW                  0

#define LWE397_ALU_IMMEDIATE_IMM2FXL_ROW                  0

#define LWE397_ALU_IMMEDIATE_IMM1FP_ROW                   0

#define LWE397_ALU_IMMEDIATE_IMM1FXH_ROW                  0

#define LWE397_ALU_IMMEDIATE_IMM1FXL_ROW                  0

#define LWE397_ALU_IMMEDIATE_IMM0FP_ROW                   0

#define LWE397_ALU_IMMEDIATE_IMM0FXH_ROW                  0

#define LWE397_ALU_IMMEDIATE_IMM0FXL_ROW                  0


// Packet LWE397_ALU_DATA_VEC
#define LWE397_ALU_DATA_VEC_SIZE 659

#define LWE397_ALU_DATA_VEC_R0_ROW                        0

#define LWE397_ALU_DATA_VEC_R1_ROW                        0

#define LWE397_ALU_DATA_VEC_R2_ROW                        0

#define LWE397_ALU_DATA_VEC_R3_ROW                        0

#define LWE397_ALU_DATA_VEC_R4_ROW                        0

#define LWE397_ALU_DATA_VEC_R5_ROW                        0

#define LWE397_ALU_DATA_VEC_R6_ROW                        0

#define LWE397_ALU_DATA_VEC_R7_ROW                        0

#define LWE397_ALU_DATA_VEC_R8_ROW                        0

#define LWE397_ALU_DATA_VEC_R9_ROW                        0

#define LWE397_ALU_DATA_VEC_R10_ROW                       0

#define LWE397_ALU_DATA_VEC_R11_ROW                       0

#define LWE397_ALU_DATA_VEC_R12_ROW                       0

#define LWE397_ALU_DATA_VEC_R13_ROW                       0

#define LWE397_ALU_DATA_VEC_R14_ROW                       0

#define LWE397_ALU_DATA_VEC_R15_ROW                       0

#define LWE397_ALU_DATA_VEC_S0_ROW                        0

#define LWE397_ALU_DATA_VEC_S1_ROW                        0

#define LWE397_ALU_DATA_VEC_S2_ROW                        0

#define LWE397_ALU_DATA_VEC_S3_ROW                        0

#define LWE397_ALU_DATA_VEC_S4_ROW                        0

#define LWE397_ALU_DATA_VEC_S5_ROW                        0

#define LWE397_ALU_DATA_VEC_S6_ROW                        0

#define LWE397_ALU_DATA_VEC_S7_ROW                        0

#define LWE397_ALU_DATA_VEC_G0_ROW                        0

#define LWE397_ALU_DATA_VEC_G1_ROW                        0

#define LWE397_ALU_DATA_VEC_G2_ROW                        0

#define LWE397_ALU_DATA_VEC_G3_ROW                        0

#define LWE397_ALU_DATA_VEC_G4_ROW                        0

#define LWE397_ALU_DATA_VEC_G5_ROW                        0

#define LWE397_ALU_DATA_VEC_G6_ROW                        0

#define LWE397_ALU_DATA_VEC_G7_ROW                        0

#define LWE397_ALU_DATA_VEC_P0_ROW                        0

#define LWE397_ALU_DATA_VEC_P1_ROW                        0

#define LWE397_ALU_DATA_VEC_P2_ROW                        0

#define LWE397_ALU_DATA_VEC_P3_ROW                        0

#define LWE397_ALU_DATA_VEC_P4_ROW                        0

#define LWE397_ALU_DATA_VEC_P5_ROW                        0

#define LWE397_ALU_DATA_VEC_P6_ROW                        0

#define LWE397_ALU_DATA_VEC_P7_ROW                        0

#define LWE397_ALU_DATA_VEC_P8_ROW                        0

#define LWE397_ALU_DATA_VEC_P9_ROW                        0

#define LWE397_ALU_DATA_VEC_P10_ROW                       0

#define LWE397_ALU_DATA_VEC_P11_ROW                       0

#define LWE397_ALU_DATA_VEC_P12_ROW                       0

#define LWE397_ALU_DATA_VEC_P13_ROW                       0

#define LWE397_ALU_DATA_VEC_P14_ROW                       0

#define LWE397_ALU_DATA_VEC_P15_ROW                       0

#define LWE397_ALU_DATA_VEC_KC_ROW                        0

#define LWE397_ALU_DATA_VEC_R0_15_ROW                     0

#define LWE397_ALU_DATA_VEC_S0_7_ROW                      0


// Packet LWE397_ALU_DATA_WE
#define LWE397_ALU_DATA_WE_SIZE 81

#define LWE397_ALU_DATA_WE_R0_LO_ROW                      0

#define LWE397_ALU_DATA_WE_R0_HI_ROW                      0

#define LWE397_ALU_DATA_WE_R1_LO_ROW                      0

#define LWE397_ALU_DATA_WE_R1_HI_ROW                      0

#define LWE397_ALU_DATA_WE_R2_LO_ROW                      0

#define LWE397_ALU_DATA_WE_R2_HI_ROW                      0

#define LWE397_ALU_DATA_WE_R3_LO_ROW                      0

#define LWE397_ALU_DATA_WE_R3_HI_ROW                      0

#define LWE397_ALU_DATA_WE_R4_LO_ROW                      0

#define LWE397_ALU_DATA_WE_R4_HI_ROW                      0

#define LWE397_ALU_DATA_WE_R5_LO_ROW                      0

#define LWE397_ALU_DATA_WE_R5_HI_ROW                      0

#define LWE397_ALU_DATA_WE_R6_LO_ROW                      0

#define LWE397_ALU_DATA_WE_R6_HI_ROW                      0

#define LWE397_ALU_DATA_WE_R7_LO_ROW                      0

#define LWE397_ALU_DATA_WE_R7_HI_ROW                      0

#define LWE397_ALU_DATA_WE_R8_LO_ROW                      0

#define LWE397_ALU_DATA_WE_R8_HI_ROW                      0

#define LWE397_ALU_DATA_WE_R9_LO_ROW                      0

#define LWE397_ALU_DATA_WE_R9_HI_ROW                      0

#define LWE397_ALU_DATA_WE_R10_LO_ROW                     0

#define LWE397_ALU_DATA_WE_R10_HI_ROW                     0

#define LWE397_ALU_DATA_WE_R11_LO_ROW                     0

#define LWE397_ALU_DATA_WE_R11_HI_ROW                     0

#define LWE397_ALU_DATA_WE_R12_LO_ROW                     0

#define LWE397_ALU_DATA_WE_R12_HI_ROW                     0

#define LWE397_ALU_DATA_WE_R13_LO_ROW                     0

#define LWE397_ALU_DATA_WE_R13_HI_ROW                     0

#define LWE397_ALU_DATA_WE_R14_LO_ROW                     0

#define LWE397_ALU_DATA_WE_R14_HI_ROW                     0

#define LWE397_ALU_DATA_WE_R15_LO_ROW                     0

#define LWE397_ALU_DATA_WE_R15_HI_ROW                     0

#define LWE397_ALU_DATA_WE_S0_LO_ROW                      0

#define LWE397_ALU_DATA_WE_S0_HI_ROW                      0

#define LWE397_ALU_DATA_WE_S1_LO_ROW                      0

#define LWE397_ALU_DATA_WE_S1_HI_ROW                      0

#define LWE397_ALU_DATA_WE_S2_LO_ROW                      0

#define LWE397_ALU_DATA_WE_S2_HI_ROW                      0

#define LWE397_ALU_DATA_WE_S3_LO_ROW                      0

#define LWE397_ALU_DATA_WE_S3_HI_ROW                      0

#define LWE397_ALU_DATA_WE_S4_LO_ROW                      0

#define LWE397_ALU_DATA_WE_S4_HI_ROW                      0

#define LWE397_ALU_DATA_WE_S5_LO_ROW                      0

#define LWE397_ALU_DATA_WE_S5_HI_ROW                      0

#define LWE397_ALU_DATA_WE_S6_LO_ROW                      0

#define LWE397_ALU_DATA_WE_S6_HI_ROW                      0

#define LWE397_ALU_DATA_WE_S7_LO_ROW                      0

#define LWE397_ALU_DATA_WE_S7_HI_ROW                      0

#define LWE397_ALU_DATA_WE_G0_LO_ROW                      0

#define LWE397_ALU_DATA_WE_G0_HI_ROW                      0

#define LWE397_ALU_DATA_WE_G1_LO_ROW                      0

#define LWE397_ALU_DATA_WE_G1_HI_ROW                      0

#define LWE397_ALU_DATA_WE_G2_LO_ROW                      0

#define LWE397_ALU_DATA_WE_G2_HI_ROW                      0

#define LWE397_ALU_DATA_WE_G3_LO_ROW                      0

#define LWE397_ALU_DATA_WE_G3_HI_ROW                      0

#define LWE397_ALU_DATA_WE_G4_LO_ROW                      0

#define LWE397_ALU_DATA_WE_G4_HI_ROW                      0

#define LWE397_ALU_DATA_WE_G5_LO_ROW                      0

#define LWE397_ALU_DATA_WE_G5_HI_ROW                      0

#define LWE397_ALU_DATA_WE_G6_LO_ROW                      0

#define LWE397_ALU_DATA_WE_G6_HI_ROW                      0

#define LWE397_ALU_DATA_WE_G7_LO_ROW                      0

#define LWE397_ALU_DATA_WE_G7_HI_ROW                      0

#define LWE397_ALU_DATA_WE_P0_ROW                 0

#define LWE397_ALU_DATA_WE_P1_ROW                 0

#define LWE397_ALU_DATA_WE_P2_ROW                 0

#define LWE397_ALU_DATA_WE_P3_ROW                 0

#define LWE397_ALU_DATA_WE_P4_ROW                 0

#define LWE397_ALU_DATA_WE_P5_ROW                 0

#define LWE397_ALU_DATA_WE_P6_ROW                 0

#define LWE397_ALU_DATA_WE_P7_ROW                 0

#define LWE397_ALU_DATA_WE_P8_ROW                 0

#define LWE397_ALU_DATA_WE_P9_ROW                 0

#define LWE397_ALU_DATA_WE_P10_ROW                        0

#define LWE397_ALU_DATA_WE_P11_ROW                        0

#define LWE397_ALU_DATA_WE_P12_ROW                        0

#define LWE397_ALU_DATA_WE_P13_ROW                        0

#define LWE397_ALU_DATA_WE_P14_ROW                        0

#define LWE397_ALU_DATA_WE_P15_ROW                        0

#define LWE397_ALU_DATA_WE_KC_ROW                 0

#define LWE397_ALU_DATA_WE_R0_15_ROW                      0

#define LWE397_ALU_DATA_WE_S0_7_ROW                       0

/*
// Packet ST_ARGS
#define ST_ARGS_SIZE 16

#define ST_ARGS_STENCIL_EN_ROW                  0
#define ST_ARGS_STENCIL_EN_DISABLE                      (0)
#define ST_ARGS_STENCIL_EN_ENABLE                       (1)

#define ST_ARGS_LOP_EN_ROW                      0
#define ST_ARGS_LOP_EN_DISABLE                  (0)
#define ST_ARGS_LOP_EN_ENABLE                   (1)

#define ST_ARGS_UPPER_ROW                       0

#define ST_ARGS_CONDITION_SELECT_ROW                    0
#define ST_ARGS_CONDITION_SELECT_ALWAYS                 (0)
#define ST_ARGS_CONDITION_SELECT_Z_WRITE                        (1)
#define ST_ARGS_CONDITION_SELECT_SURFACE_WRITE                  (2)

#define ST_ARGS_RESERVED_ROW                    0


// Packet ST_R20_ARGS
#define ST_R20_ARGS_SIZE 16

#define ST_R20_ARGS_REG_ROW                     0
#define ST_R20_ARGS_REG_R0                      (0)
#define ST_R20_ARGS_REG_R1                      (1)
#define ST_R20_ARGS_REG_R2                      (2)
#define ST_R20_ARGS_REG_R3                      (3)

#define ST_R20_ARGS_WRITE_MASK_ROW                      0

#define ST_R20_ARGS_OFFSET_IMM_ROW                      0

#define ST_R20_ARGS_OFFSET_REG_ROW                      0
#define ST_R20_ARGS_OFFSET_REG_R0                       (0)
#define ST_R20_ARGS_OFFSET_REG_R1                       (1)
#define ST_R20_ARGS_OFFSET_REG_R2                       (2)
#define ST_R20_ARGS_OFFSET_REG_R3                       (3)

#define ST_R20_ARGS_OFFSET_MOD_ROW                      0
#define ST_R20_ARGS_OFFSET_MOD_L                        (0)
#define ST_R20_ARGS_OFFSET_MOD_H                        (1)

#define ST_R20_ARGS_OFFSET_REG_EN_ROW                   0
#define ST_R20_ARGS_OFFSET_REG_EN_DISABLE                       (0)
#define ST_R20_ARGS_OFFSET_REG_EN_ENABLE                        (1)

#define ST_R20_ARGS_OFFSET_TYPE_ROW                     0
#define ST_R20_ARGS_OFFSET_TYPE_LOCAL                   (0)
#define ST_R20_ARGS_OFFSET_TYPE_GLOBAL                  (1)


// Packet ST_R80_ARGS
#define ST_R80_ARGS_SIZE 16

#define ST_R80_ARGS_WRITE_MASK_ROW                      0

#define ST_R80_ARGS_OFFSET_IMM_ROW                      0

#define ST_R80_ARGS_OFFSET_REG_ROW                      0
#define ST_R80_ARGS_OFFSET_REG_R0                       (0)
#define ST_R80_ARGS_OFFSET_REG_R1                       (1)
#define ST_R80_ARGS_OFFSET_REG_R2                       (2)
#define ST_R80_ARGS_OFFSET_REG_R3                       (3)

#define ST_R80_ARGS_OFFSET_MOD_ROW                      0
#define ST_R80_ARGS_OFFSET_MOD_L                        (0)
#define ST_R80_ARGS_OFFSET_MOD_H                        (1)

#define ST_R80_ARGS_OFFSET_REG_EN_ROW                   0
#define ST_R80_ARGS_OFFSET_REG_EN_DISABLE                       (0)
#define ST_R80_ARGS_OFFSET_REG_EN_ENABLE                        (1)

#define ST_R80_ARGS_OFFSET_TYPE_ROW                     0
#define ST_R80_ARGS_OFFSET_TYPE_LOCAL                   (0)
#define ST_R80_ARGS_OFFSET_TYPE_GLOBAL                  (1)
*/

// Register LWE397_DW_INST_OFFSET_0  
#define LWE397_DW_INST_OFFSET_0                   (0x900)
#define LWE397_DW_INST_OFFSET_0_INDEX                       5:0


// Register LWE397_DW_INST_DATA_0  
#define LWE397_DW_INST_DATA_0                     (0x901)
#define LWE397_DW_INST_DATA_0_OP                    1:0
#define LWE397_DW_INST_DATA_0_OP_NOP                      (0)
#define LWE397_DW_INST_DATA_0_OP_ST                       (1)
#define LWE397_DW_INST_DATA_0_OP_ST_R20                   (2)
#define LWE397_DW_INST_DATA_0_OP_ST_R80                   (3)

#define LWE397_DW_INST_DATA_0_SURF                  5:2

#define LWE397_DW_INST_DATA_0_WRITE_KILLED                  6:6
#define LWE397_DW_INST_DATA_0_WRITE_KILLED_DISABLE                        (0)
#define LWE397_DW_INST_DATA_0_WRITE_KILLED_ENABLE                 (1)

#define LWE397_DW_INST_DATA_0_WRITE_NON_CENTER                      7:7
#define LWE397_DW_INST_DATA_0_WRITE_NON_CENTER_DISABLE                    (0)
#define LWE397_DW_INST_DATA_0_WRITE_NON_CENTER_ENABLE                     (1)

#define LWE397_DW_INST_DATA_0_CACHE_PERSISTENT                      8:8

#define LWE397_DW_INST_DATA_0_CACHE_READ_CLEAN                      9:9

#define LWE397_DW_INST_DATA_0_ARGS                  25:10


// Register LWE397_DW_LOGIC_OP_0  
#define LWE397_DW_LOGIC_OP_0                      (0x902)
#define LWE397_DW_LOGIC_OP_0_OP                     3:0
#define LWE397_DW_LOGIC_OP_0_OP_CLEAR                     (0)
#define LWE397_DW_LOGIC_OP_0_OP_AND                       (1)
#define LWE397_DW_LOGIC_OP_0_OP_AND_REVERSE                       (2)
#define LWE397_DW_LOGIC_OP_0_OP_COPY                      (3)
#define LWE397_DW_LOGIC_OP_0_OP_AND_ILWERTED                      (4)
#define LWE397_DW_LOGIC_OP_0_OP_NOOP                      (5)
#define LWE397_DW_LOGIC_OP_0_OP_XOR                       (6)
#define LWE397_DW_LOGIC_OP_0_OP_OR                        (7)
#define LWE397_DW_LOGIC_OP_0_OP_NOR                       (8)
#define LWE397_DW_LOGIC_OP_0_OP_EQUIV                     (9)
#define LWE397_DW_LOGIC_OP_0_OP_ILWERT                    (10)
#define LWE397_DW_LOGIC_OP_0_OP_OR_REVERSE                        (11)
#define LWE397_DW_LOGIC_OP_0_OP_COPY_ILWERTED                     (12)
#define LWE397_DW_LOGIC_OP_0_OP_OR_ILWERTED                       (13)
#define LWE397_DW_LOGIC_OP_0_OP_NAND                      (14)
#define LWE397_DW_LOGIC_OP_0_OP_SET                       (15)


// Register LWE397_DW_ST_ENABLE_0  
#define LWE397_DW_ST_ENABLE_0                     (0x903)
#define LWE397_DW_ST_ENABLE_0_SURFACE_0_ST_ENABLE                   0:0
#define LWE397_DW_ST_ENABLE_0_SURFACE_0_ST_ENABLE_DISABLE                 (0)
#define LWE397_DW_ST_ENABLE_0_SURFACE_0_ST_ENABLE_ENABLE                  (1)

#define LWE397_DW_ST_ENABLE_0_SURFACE_1_ST_ENABLE                   1:1
#define LWE397_DW_ST_ENABLE_0_SURFACE_1_ST_ENABLE_DISABLE                 (0)
#define LWE397_DW_ST_ENABLE_0_SURFACE_1_ST_ENABLE_ENABLE                  (1)

#define LWE397_DW_ST_ENABLE_0_SURFACE_2_ST_ENABLE                   2:2
#define LWE397_DW_ST_ENABLE_0_SURFACE_2_ST_ENABLE_DISABLE                 (0)
#define LWE397_DW_ST_ENABLE_0_SURFACE_2_ST_ENABLE_ENABLE                  (1)

#define LWE397_DW_ST_ENABLE_0_SURFACE_3_ST_ENABLE                   3:3
#define LWE397_DW_ST_ENABLE_0_SURFACE_3_ST_ENABLE_DISABLE                 (0)
#define LWE397_DW_ST_ENABLE_0_SURFACE_3_ST_ENABLE_ENABLE                  (1)

#define LWE397_DW_ST_ENABLE_0_SURFACE_4_ST_ENABLE                   4:4
#define LWE397_DW_ST_ENABLE_0_SURFACE_4_ST_ENABLE_DISABLE                 (0)
#define LWE397_DW_ST_ENABLE_0_SURFACE_4_ST_ENABLE_ENABLE                  (1)

#define LWE397_DW_ST_ENABLE_0_SURFACE_5_ST_ENABLE                   5:5
#define LWE397_DW_ST_ENABLE_0_SURFACE_5_ST_ENABLE_DISABLE                 (0)
#define LWE397_DW_ST_ENABLE_0_SURFACE_5_ST_ENABLE_ENABLE                  (1)

#define LWE397_DW_ST_ENABLE_0_SURFACE_6_ST_ENABLE                   6:6
#define LWE397_DW_ST_ENABLE_0_SURFACE_6_ST_ENABLE_DISABLE                 (0)
#define LWE397_DW_ST_ENABLE_0_SURFACE_6_ST_ENABLE_ENABLE                  (1)

#define LWE397_DW_ST_ENABLE_0_SURFACE_7_ST_ENABLE                   7:7
#define LWE397_DW_ST_ENABLE_0_SURFACE_7_ST_ENABLE_DISABLE                 (0)
#define LWE397_DW_ST_ENABLE_0_SURFACE_7_ST_ENABLE_ENABLE                  (1)

#define LWE397_DW_ST_ENABLE_0_SURFACE_8_ST_ENABLE                   8:8
#define LWE397_DW_ST_ENABLE_0_SURFACE_8_ST_ENABLE_DISABLE                 (0)
#define LWE397_DW_ST_ENABLE_0_SURFACE_8_ST_ENABLE_ENABLE                  (1)

#define LWE397_DW_ST_ENABLE_0_SURFACE_9_ST_ENABLE                   9:9
#define LWE397_DW_ST_ENABLE_0_SURFACE_9_ST_ENABLE_DISABLE                 (0)
#define LWE397_DW_ST_ENABLE_0_SURFACE_9_ST_ENABLE_ENABLE                  (1)

#define LWE397_DW_ST_ENABLE_0_SURFACE_10_ST_ENABLE                  10:10
#define LWE397_DW_ST_ENABLE_0_SURFACE_10_ST_ENABLE_DISABLE                        (0)
#define LWE397_DW_ST_ENABLE_0_SURFACE_10_ST_ENABLE_ENABLE                 (1)

#define LWE397_DW_ST_ENABLE_0_SURFACE_11_ST_ENABLE                  11:11
#define LWE397_DW_ST_ENABLE_0_SURFACE_11_ST_ENABLE_DISABLE                        (0)
#define LWE397_DW_ST_ENABLE_0_SURFACE_11_ST_ENABLE_ENABLE                 (1)

#define LWE397_DW_ST_ENABLE_0_SURFACE_12_ST_ENABLE                  12:12
#define LWE397_DW_ST_ENABLE_0_SURFACE_12_ST_ENABLE_DISABLE                        (0)
#define LWE397_DW_ST_ENABLE_0_SURFACE_12_ST_ENABLE_ENABLE                 (1)

#define LWE397_DW_ST_ENABLE_0_SURFACE_13_ST_ENABLE                  13:13
#define LWE397_DW_ST_ENABLE_0_SURFACE_13_ST_ENABLE_DISABLE                        (0)
#define LWE397_DW_ST_ENABLE_0_SURFACE_13_ST_ENABLE_ENABLE                 (1)

#define LWE397_DW_ST_ENABLE_0_SURFACE_14_ST_ENABLE                  14:14
#define LWE397_DW_ST_ENABLE_0_SURFACE_14_ST_ENABLE_DISABLE                        (0)
#define LWE397_DW_ST_ENABLE_0_SURFACE_14_ST_ENABLE_ENABLE                 (1)

#define LWE397_DW_ST_ENABLE_0_SURFACE_15_ST_ENABLE                  15:15
#define LWE397_DW_ST_ENABLE_0_SURFACE_15_ST_ENABLE_DISABLE                        (0)
#define LWE397_DW_ST_ENABLE_0_SURFACE_15_ST_ENABLE_ENABLE                 (1)

#define LWE397_DW_ST_ENABLE_0_SURFACES_ENABLE                       15:0

#define LWE397_DW_ST_ENABLE_0_Z_TEST                        16:16
#define LWE397_DW_ST_ENABLE_0_Z_TEST_DISABLE                      (0)
#define LWE397_DW_ST_ENABLE_0_Z_TEST_ENABLE                       (1)


// Register LWE397_DW_MEMORY_OUTPUT_ADDRESS_0  
#define LWE397_DW_MEMORY_OUTPUT_ADDRESS_0                 (0x904)
#define LWE397_DW_MEMORY_OUTPUT_ADDRESS_0_BASE_ADDRESS                      31:0


// Register LWE397_DW_MEMORY_OUTPUT_DATA_0  
#define LWE397_DW_MEMORY_OUTPUT_DATA_0                    (0x905)
#define LWE397_DW_MEMORY_OUTPUT_DATA_0_DATA                 31:0


// Register LWE397_DW_MEMORY_OUTPUT_INCR_0  
#define LWE397_DW_MEMORY_OUTPUT_INCR_0                    (0x906)
#define LWE397_DW_MEMORY_OUTPUT_INCR_0_AMOUNT                       0:0


// Register LWE397_DW_TIMESTAMP_CTL_0  
#define LWE397_DW_TIMESTAMP_CTL_0                 (0x907)
#define LWE397_DW_TIMESTAMP_CTL_0_ENABLE                    0:0


// Register LWE397_DW_TIMESTAMP_LOW_0  
#define LWE397_DW_TIMESTAMP_LOW_0                 (0x908)
#define LWE397_DW_TIMESTAMP_LOW_0_VALUE                     31:0


// Register LWE397_DW_TIMESTAMP_HIGH_0  
#define LWE397_DW_TIMESTAMP_HIGH_0                        (0x909)
#define LWE397_DW_TIMESTAMP_HIGH_0_VALUE                    31:0


// Register LWE397_DW_PIXEL_COUNT_CTRL_0  
#define LWE397_DW_PIXEL_COUNT_CTRL_0                      (0x90a)
#define LWE397_DW_PIXEL_COUNT_CTRL_0_ENABLE                 0:0

#define LWE397_DW_PIXEL_COUNT_CTRL_0_COUNT_NON_CENTER                       1:1


// Register LWE397_DW_PIXEL_COUNT_0  
#define LWE397_DW_PIXEL_COUNT_0                   (0x90b)
#define LWE397_DW_PIXEL_COUNT_0_VALUE                       31:0

#define LW_FDC_MAX_LINES_CNT_WIDTH      9

// Register LWE397_FDC_CONTROL_0  
#define LWE397_FDC_CONTROL_0                      (0xa00)
#define LWE397_FDC_CONTROL_0_ILWALIDATE                     0:0

#define LWE397_FDC_CONTROL_0_FLUSH                  1:1

#define LWE397_FDC_CONTROL_0_STRICT_RI_ARB                  2:2

#define LWE397_FDC_CONTROL_0_STRICT_L2_ARB                  3:3

#define LWE397_FDC_CONTROL_0_DISABLE_AUTO_FLUSH                     4:4

#define LWE397_FDC_CONTROL_0_PERSISTENT_CLEAR                       5:5

#define LWE397_FDC_CONTROL_0_STRICT_MC_WR_ARB                       6:6

#define LWE397_FDC_CONTROL_0_DISABLE_PERSISTENT                     7:7

#define LWE397_FDC_CONTROL_0_DISABLE_READ_CLEAN                     8:8

#define LWE397_FDC_CONTROL_0_STRICT_DP_Q_RI_ARB                     9:9

#define LWE397_FDC_CONTROL_0_MCCIF_PORT_ASSIGN                      10:10
#define LWE397_FDC_CONTROL_0_MCCIF_PORT_ASSIGN_EXT_INT                    (0)
#define LWE397_FDC_CONTROL_0_MCCIF_PORT_ASSIGN_EVEN_ODD                   (1)

#define LWE397_FDC_CONTROL_0_READ_BUF_MIN_LINE                      11:11

#define LWE397_FDC_CONTROL_0_FORCE_PIPE_CLEAN                       12:12


// Register LWE397_FDC_STATUS_0  
#define LWE397_FDC_STATUS_0                       (0xa01)
#define LWE397_FDC_STATUS_0_FLUSH_DONE                      0:0

#define LWE397_FDC_STATUS_0_CSIM_SHUTDOWN_STATUS                    31:31


// Register LWE397_FDC_MAX_QZ_LINES_0  
#define LWE397_FDC_MAX_QZ_LINES_0                 (0xa02)
#define LWE397_FDC_MAX_QZ_LINES_0_MAX_QZ_LINES                      8:0


// Register LWE397_FDC_MAX_QV_LINES_0  
#define LWE397_FDC_MAX_QV_LINES_0                 (0xa03)
#define LWE397_FDC_MAX_QV_LINES_0_MAX_QV_LINES                      8:0


// Register LWE397_FDC_MAX_QS_LINES_0  
#define LWE397_FDC_MAX_QS_LINES_0                 (0xa04)
#define LWE397_FDC_MAX_QS_LINES_0_MAX_QS_LINES                      8:0


// Register LWE397_FDC_MAX_PS_LINES_0  
#define LWE397_FDC_MAX_PS_LINES_0                 (0xa05)
#define LWE397_FDC_MAX_PS_LINES_0_MAX_PS_LINES                      8:0


// Register LWE397_FDC_MAX_Q_LINES_0  
#define LWE397_FDC_MAX_Q_LINES_0                  (0xa06)
#define LWE397_FDC_MAX_Q_LINES_0_MAX_Q_LINES                        8:0


// Register LWE397_FDC_MAX_Q_P_LINES_0  
#define LWE397_FDC_MAX_Q_P_LINES_0                        (0xa07)
#define LWE397_FDC_MAX_Q_P_LINES_0_MAX_Q_P_LINES                    8:0


// Register LWE397_FDC_FLUSH_CTL_0  
#define LWE397_FDC_FLUSH_CTL_0                    (0xa08)
#define LWE397_FDC_FLUSH_CTL_0_DISABLE_AGGRESSIVE_FLUSH1                    17:17

#define LWE397_FDC_FLUSH_CTL_0_DISABLE_AGGRESSIVE_FLUSH0                    16:16

#define LWE397_FDC_FLUSH_CTL_0_MAX_FLUSH_STALLS                     12:8

#define LWE397_FDC_FLUSH_CTL_0_AUTO_FLUSH_DELAY                     7:0

#define LW_FDC_L1_TIMEOUT_CNT_W 5

// Register LWE397_FDC_L1_TIMEOUT_0  
#define LWE397_FDC_L1_TIMEOUT_0                   (0xa09)
#define LWE397_FDC_L1_TIMEOUT_0_D_TIMEOUT                   28:24

#define LWE397_FDC_L1_TIMEOUT_0_QW_TIMEOUT                  20:16

#define LWE397_FDC_L1_TIMEOUT_0_P_TIMEOUT                   12:8

#define LWE397_FDC_L1_TIMEOUT_0_QR_TIMEOUT                  4:0


// Register LWE397_FDC_INSTRUMENT_0  
#define LWE397_FDC_INSTRUMENT_0                   (0xa0a)
#define LWE397_FDC_INSTRUMENT_0_STAT_EN                     0:0


// Register LWE397_FDC_CLKEN_OVERRIDE_0  
#define LWE397_FDC_CLKEN_OVERRIDE_0                       (0xa0b)
#define LWE397_FDC_CLKEN_OVERRIDE_0_FDC_CLKEN_OVR                   0:0
#define LWE397_FDC_CLKEN_OVERRIDE_0_FDC_CLKEN_OVR_CLK_GATED                       (0)
#define LWE397_FDC_CLKEN_OVERRIDE_0_FDC_CLKEN_OVR_CLK_ALWAYS_ON                   (1)

#define LWE397_FDC_CLKEN_OVERRIDE_0_FDCPR_CLKEN_OVR                 1:1
#define LWE397_FDC_CLKEN_OVERRIDE_0_FDCPR_CLKEN_OVR_CLK_GATED                     (0)
#define LWE397_FDC_CLKEN_OVERRIDE_0_FDCPR_CLKEN_OVR_CLK_ALWAYS_ON                 (1)

#define LWE397_FDC_CLKEN_OVERRIDE_0_FDCZR_CLKEN_OVR                 2:2
#define LWE397_FDC_CLKEN_OVERRIDE_0_FDCZR_CLKEN_OVR_CLK_GATED                     (0)
#define LWE397_FDC_CLKEN_OVERRIDE_0_FDCZR_CLKEN_OVR_CLK_ALWAYS_ON                 (1)

#define LWE397_FDC_CLKEN_OVERRIDE_0_FDCSR_CLKEN_OVR                 3:3
#define LWE397_FDC_CLKEN_OVERRIDE_0_FDCSR_CLKEN_OVR_CLK_GATED                     (0)
#define LWE397_FDC_CLKEN_OVERRIDE_0_FDCSR_CLKEN_OVR_CLK_ALWAYS_ON                 (1)

#define LWE397_FDC_CLKEN_OVERRIDE_0_FDCVR_CLKEN_OVR                 4:4
#define LWE397_FDC_CLKEN_OVERRIDE_0_FDCVR_CLKEN_OVR_CLK_GATED                     (0)
#define LWE397_FDC_CLKEN_OVERRIDE_0_FDCVR_CLKEN_OVR_CLK_ALWAYS_ON                 (1)

#define LWE397_FDC_CLKEN_OVERRIDE_0_FDCDW_CLKEN_OVR                 5:5
#define LWE397_FDC_CLKEN_OVERRIDE_0_FDCDW_CLKEN_OVR_CLK_GATED                     (0)
#define LWE397_FDC_CLKEN_OVERRIDE_0_FDCDW_CLKEN_OVR_CLK_ALWAYS_ON                 (1)

#define LWE397_FDC_CLKEN_OVERRIDE_0_FDCZW_CLKEN_OVR                 6:6
#define LWE397_FDC_CLKEN_OVERRIDE_0_FDCZW_CLKEN_OVR_CLK_GATED                     (0)
#define LWE397_FDC_CLKEN_OVERRIDE_0_FDCZW_CLKEN_OVR_CLK_ALWAYS_ON                 (1)

#define LWE397_FDC_CLKEN_OVERRIDE_0_FDCSW_CLKEN_OVR                 7:7
#define LWE397_FDC_CLKEN_OVERRIDE_0_FDCSW_CLKEN_OVR_CLK_GATED                     (0)
#define LWE397_FDC_CLKEN_OVERRIDE_0_FDCSW_CLKEN_OVR_CLK_ALWAYS_ON                 (1)

#define LWE397_FDC_CLKEN_OVERRIDE_0_FDCVW_CLKEN_OVR                 8:8
#define LWE397_FDC_CLKEN_OVERRIDE_0_FDCVW_CLKEN_OVR_CLK_GATED                     (0)
#define LWE397_FDC_CLKEN_OVERRIDE_0_FDCVW_CLKEN_OVR_CLK_ALWAYS_ON                 (1)

#define LWE397_FDC_CLKEN_OVERRIDE_0_FDCSB_CLKEN_OVR                 9:9
#define LWE397_FDC_CLKEN_OVERRIDE_0_FDCSB_CLKEN_OVR_CLK_GATED                     (0)
#define LWE397_FDC_CLKEN_OVERRIDE_0_FDCSB_CLKEN_OVR_CLK_ALWAYS_ON                 (1)


// Register LWE397_FDC_LW_MCCIF_FIFOCTRL_RO_0  
#define LWE397_FDC_LW_MCCIF_FIFOCTRL_RO_0                 (0xa0c)
#define LWE397_FDC_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_WRCL_MCLE2X                      0:0
#define LWE397_FDC_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_WRCL_MCLE2X_DISABLE                    (0)
#define LWE397_FDC_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_WRCL_MCLE2X_ENABLE                     (1)

#define LWE397_FDC_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_RDMC_RDFAST                      1:1
#define LWE397_FDC_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_RDMC_RDFAST_DISABLE                    (0)
#define LWE397_FDC_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_RDMC_RDFAST_ENABLE                     (1)

#define LWE397_FDC_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_WRMC_CLLE2X                      2:2
#define LWE397_FDC_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_WRMC_CLLE2X_DISABLE                    (0)
#define LWE397_FDC_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_WRMC_CLLE2X_ENABLE                     (1)

#define LWE397_FDC_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_RDCL_RDFAST                      3:3
#define LWE397_FDC_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_RDCL_RDFAST_DISABLE                    (0)
#define LWE397_FDC_LW_MCCIF_FIFOCTRL_RO_0_LW_MCCIF_RDCL_RDFAST_ENABLE                     (1)

#define LWE397_FDC_LW_MCCIF_FIFOCTRL_RO_0_LW_WCLK_OVERRIDE                  16:16

#define LWE397_FDC_LW_MCCIF_FIFOCTRL_RO_0_LW_RCLK_OVERRIDE                  17:17

#define LWE397_ILWALID_CHANNEL    15

// Register LWE397_GSHIM_WRITE_MASK_0  
#define LWE397_GSHIM_WRITE_MASK_0                 (0xb00)
#define LWE397_GSHIM_WRITE_MASK_0_GPU_A                     0:0
#define LWE397_GSHIM_WRITE_MASK_0_GPU_A_DISABLE                   (0)
#define LWE397_GSHIM_WRITE_MASK_0_GPU_A_ENABLE                    (1)

#define LWE397_GSHIM_WRITE_MASK_0_GPU_B                     1:1
#define LWE397_GSHIM_WRITE_MASK_0_GPU_B_DISABLE                   (0)
#define LWE397_GSHIM_WRITE_MASK_0_GPU_B_ENABLE                    (1)


// Register LWE397_GSHIM_READ_SELECT_0  
#define LWE397_GSHIM_READ_SELECT_0                        (0xb01)
#define LWE397_GSHIM_READ_SELECT_0_GPU                      1:0
#define LWE397_GSHIM_READ_SELECT_0_GPU_GPU_A                      (0)
#define LWE397_GSHIM_READ_SELECT_0_GPU_GPU_B                      (1)
#define LWE397_GSHIM_READ_SELECT_0_GPU_IMMEDIATE                  (2)


// Register LWE397_GSHIM_FLUSH_0  
#define LWE397_GSHIM_FLUSH_0                      (0xb02)
#define LWE397_GSHIM_FLUSH_0_FLUSH                  0:0


// Register LWE397_GSHIM_SCISSOR_SPLIT_0  
#define LWE397_GSHIM_SCISSOR_SPLIT_0                      (0xb03)
#define LWE397_GSHIM_SCISSOR_SPLIT_0_SPLIT_AXIS                     0:0
#define LWE397_GSHIM_SCISSOR_SPLIT_0_SPLIT_AXIS_X                 (0)
#define LWE397_GSHIM_SCISSOR_SPLIT_0_SPLIT_AXIS_Y                 (1)

#define LWE397_GSHIM_SCISSOR_SPLIT_0_SWAP                   1:1
#define LWE397_GSHIM_SCISSOR_SPLIT_0_SWAP_DISABLE                 (0)
#define LWE397_GSHIM_SCISSOR_SPLIT_0_SWAP_ENABLE                  (1)

#define LWE397_GSHIM_SCISSOR_SPLIT_0_VCAA                   3:3

#define LWE397_GSHIM_SCISSOR_SPLIT_0_SPLIT_POINT                    14:2


// Register LWE397_GSHIM_STAT_ENABLE_0  
#define LWE397_GSHIM_STAT_ENABLE_0                        (0xb04)
#define LWE397_GSHIM_STAT_ENABLE_0_EN                       0:0
#define LWE397_GSHIM_STAT_ENABLE_0_EN_DISABLE                     (0)
#define LWE397_GSHIM_STAT_ENABLE_0_EN_ENABLE                      (1)

#define LWE397_GSHIM_STAT_ENABLE_0_OVR                      2:2

#define LWE397_GSHIM_STAT_ENABLE_0_INDEX                    5:3


// Register LWE397_GSHIM_STAT_STALL_0  
#define LWE397_GSHIM_STAT_STALL_0                 (0xb06)
#define LWE397_GSHIM_STAT_STALL_0_COUNT                     31:0


// Register LWE397_GSHIM_STAT_STALL  
#define LWE397_GSHIM_STAT_STALL                   (0xb06)
#define LWE397_GSHIM_STAT_STALL_COUNT                       31:0


// Register LWE397_GSHIM_STAT_STALL_1  
#define LWE397_GSHIM_STAT_STALL_1                 (0xb07)
#define LWE397_GSHIM_STAT_STALL_1_COUNT                     31:0


// Register LWE397_GSHIM_STAT_WAIT_0  
#define LWE397_GSHIM_STAT_WAIT_0                  (0xb08)
#define LWE397_GSHIM_STAT_WAIT_0_COUNT                      31:0


// Register LWE397_GSHIM_STAT_WAIT  
#define LWE397_GSHIM_STAT_WAIT                    (0xb08)
#define LWE397_GSHIM_STAT_WAIT_COUNT                        31:0


// Register LWE397_GSHIM_STAT_WAIT_1  
#define LWE397_GSHIM_STAT_WAIT_1                  (0xb09)
#define LWE397_GSHIM_STAT_WAIT_1_COUNT                      31:0


// Register LWE397_GSHIM_STAT_COMB_0  
#define LWE397_GSHIM_STAT_COMB_0                  (0xb0a)
#define LWE397_GSHIM_STAT_COMB_0_COUNT                      31:0


// Register LWE397_GSHIM_STAT_COMB  
#define LWE397_GSHIM_STAT_COMB                    (0xb0a)
#define LWE397_GSHIM_STAT_COMB_COUNT                        31:0


// Register LWE397_GSHIM_STAT_COMB_1  
#define LWE397_GSHIM_STAT_COMB_1                  (0xb0b)
#define LWE397_GSHIM_STAT_COMB_1_COUNT                      31:0


// Register LWE397_GSHIM_STAT_HWR_WAIT_0  
#define LWE397_GSHIM_STAT_HWR_WAIT_0                      (0xb0c)
#define LWE397_GSHIM_STAT_HWR_WAIT_0_COUNT                  31:0


// Register LWE397_GSHIM_STAT_HWR_XFER_0  
#define LWE397_GSHIM_STAT_HWR_XFER_0                      (0xb0d)
#define LWE397_GSHIM_STAT_HWR_XFER_0_COUNT                  31:0


// Register LWE397_GSHIM_STAT_SYNCPT_WAIT_0  
#define LWE397_GSHIM_STAT_SYNCPT_WAIT_0                   (0xb0e)
#define LWE397_GSHIM_STAT_SYNCPT_WAIT_0_COUNT                       31:0


// Register LWE397_GSHIM_STAT_SYNCPT_WAIT  
#define LWE397_GSHIM_STAT_SYNCPT_WAIT                     (0xb0e)
#define LWE397_GSHIM_STAT_SYNCPT_WAIT_COUNT                 31:0


// Register LWE397_GSHIM_STAT_SYNCPT_WAIT_1  
#define LWE397_GSHIM_STAT_SYNCPT_WAIT_1                   (0xb0f)
#define LWE397_GSHIM_STAT_SYNCPT_WAIT_1_COUNT                       31:0


// Register LWE397_GSHIM_DLB_CONTROL_0  
#define LWE397_GSHIM_DLB_CONTROL_0                        (0xb10)
#define LWE397_GSHIM_DLB_CONTROL_0_MIN_INCR                 5:0

#define LWE397_GSHIM_DLB_CONTROL_0_MAX_INCR                 17:6

#define LWE397_GSHIM_DLB_CONTROL_0_HISTORY_SIZE                     24:18


// Register LWE397_GSHIM_DLB_0  
#define LWE397_GSHIM_DLB_0                  (0xb11)
#define LWE397_GSHIM_DLB_0_MIN                        12:0

#define LWE397_GSHIM_DLB_0_MAX                        26:13


// Register LWE397_GSHIM_DLB_TRIGGER_0  
#define LWE397_GSHIM_DLB_TRIGGER_0                        (0xb12)
#define LWE397_GSHIM_DLB_TRIGGER_0_EVENT                    1:0
#define LWE397_GSHIM_DLB_TRIGGER_0_EVENT_NEVER                    (0)
#define LWE397_GSHIM_DLB_TRIGGER_0_EVENT_NOW                      (1)
#define LWE397_GSHIM_DLB_TRIGGER_0_EVENT_IDLE                     (2)
#define LWE397_GSHIM_DLB_TRIGGER_0_EVENT_TIMEOUT                  (3)

#define LWE397_GSHIM_DLB_TRIGGER_0_TIMEOUT                  31:2


// Register LWE397_GSHIM_HIGH_WATERMARK_0  
#define LWE397_GSHIM_HIGH_WATERMARK_0                     (0xb13)
#define LWE397_GSHIM_HIGH_WATERMARK_0_COUNT                 13:0


// Register LWE397_GSHIM_DEBUG0_0  
#define LWE397_GSHIM_DEBUG0_0                     (0xb14)
#define LWE397_GSHIM_DEBUG0_0_VAL                   31:0


// Register LWE397_GLOBAL_SURFADDR_0  
#define LWE397_GLOBAL_SURFADDR_0                  (0xe00)
#define LWE397_GLOBAL_SURFADDR_0_BASE_ADDRESS                       31:0


// Register LWE397_GLOBAL_SURFADDR  
#define LWE397_GLOBAL_SURFADDR                    (0xe00)
#define LWE397_GLOBAL_SURFADDR_BASE_ADDRESS                 31:0


// Register LWE397_GLOBAL_SURFADDR_1  
#define LWE397_GLOBAL_SURFADDR_1                  (0xe01)
#define LWE397_GLOBAL_SURFADDR_1_BASE_ADDRESS                       31:0


// Register LWE397_GLOBAL_SURFADDR_2  
#define LWE397_GLOBAL_SURFADDR_2                  (0xe02)
#define LWE397_GLOBAL_SURFADDR_2_BASE_ADDRESS                       31:0


// Register LWE397_GLOBAL_SURFADDR_3  
#define LWE397_GLOBAL_SURFADDR_3                  (0xe03)
#define LWE397_GLOBAL_SURFADDR_3_BASE_ADDRESS                       31:0


// Register LWE397_GLOBAL_SURFADDR_4  
#define LWE397_GLOBAL_SURFADDR_4                  (0xe04)
#define LWE397_GLOBAL_SURFADDR_4_BASE_ADDRESS                       31:0


// Register LWE397_GLOBAL_SURFADDR_5  
#define LWE397_GLOBAL_SURFADDR_5                  (0xe05)
#define LWE397_GLOBAL_SURFADDR_5_BASE_ADDRESS                       31:0


// Register LWE397_GLOBAL_SURFADDR_6  
#define LWE397_GLOBAL_SURFADDR_6                  (0xe06)
#define LWE397_GLOBAL_SURFADDR_6_BASE_ADDRESS                       31:0


// Register LWE397_GLOBAL_SURFADDR_7  
#define LWE397_GLOBAL_SURFADDR_7                  (0xe07)
#define LWE397_GLOBAL_SURFADDR_7_BASE_ADDRESS                       31:0


// Register LWE397_GLOBAL_SURFADDR_8  
#define LWE397_GLOBAL_SURFADDR_8                  (0xe08)
#define LWE397_GLOBAL_SURFADDR_8_BASE_ADDRESS                       31:0


// Register LWE397_GLOBAL_SURFADDR_9  
#define LWE397_GLOBAL_SURFADDR_9                  (0xe09)
#define LWE397_GLOBAL_SURFADDR_9_BASE_ADDRESS                       31:0


// Register LWE397_GLOBAL_SURFADDR_10  
#define LWE397_GLOBAL_SURFADDR_10                 (0xe0a)
#define LWE397_GLOBAL_SURFADDR_10_BASE_ADDRESS                      31:0


// Register LWE397_GLOBAL_SURFADDR_11  
#define LWE397_GLOBAL_SURFADDR_11                 (0xe0b)
#define LWE397_GLOBAL_SURFADDR_11_BASE_ADDRESS                      31:0


// Register LWE397_GLOBAL_SURFADDR_12  
#define LWE397_GLOBAL_SURFADDR_12                 (0xe0c)
#define LWE397_GLOBAL_SURFADDR_12_BASE_ADDRESS                      31:0


// Register LWE397_GLOBAL_SURFADDR_13  
#define LWE397_GLOBAL_SURFADDR_13                 (0xe0d)
#define LWE397_GLOBAL_SURFADDR_13_BASE_ADDRESS                      31:0


// Register LWE397_GLOBAL_SURFADDR_14  
#define LWE397_GLOBAL_SURFADDR_14                 (0xe0e)
#define LWE397_GLOBAL_SURFADDR_14_BASE_ADDRESS                      31:0


// Register LWE397_GLOBAL_SURFADDR_15  
#define LWE397_GLOBAL_SURFADDR_15                 (0xe0f)
#define LWE397_GLOBAL_SURFADDR_15_BASE_ADDRESS                      31:0

#define LW_GR3D_SURF_W_QUANTUM  8
#define LW_GR3D_SURF_BYTE_QUANTUM       16

// Register LWE397_GLOBAL_SURFDESC_0  
#define LWE397_GLOBAL_SURFDESC_0                  (0xe10)
#define LWE397_GLOBAL_SURFDESC_0_OVERLAP                    27:27
#define LWE397_GLOBAL_SURFDESC_0_OVERLAP_DISABLE                  (0)
#define LWE397_GLOBAL_SURFDESC_0_OVERLAP_ENABLE                   (1)

#define LWE397_GLOBAL_SURFDESC_0_STRUCTURE                  26:25
#define LWE397_GLOBAL_SURFDESC_0_STRUCTURE_LINEAR                 (0)
#define LWE397_GLOBAL_SURFDESC_0_STRUCTURE_TILED                  (1)
#define LWE397_GLOBAL_SURFDESC_0_STRUCTURE_XY_TILED                       (2)

#define LWE397_GLOBAL_SURFDESC_0_ARRAY_STRIDE                       24:8

#define LWE397_GLOBAL_SURFDESC_0_STRIDE                     23:8

#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT                        7:2
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_C4X4                 (0)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_A8                   (1)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_L8                   (2)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_S8                   (3)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_L8A8                 (4)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_B2G3R3                       (5)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_B5G6R5                       (6)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_B5G5R5A1                     (7)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_B4G4R4A4                     (8)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_A1B5G5R5                     (9)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_A4B4G4R4                     (10)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_Z16                  (11)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_Z16NL                        (12)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_R8G8B8A8                     (13)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_B8G8R8A8                     (14)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_A16_float                    (15)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_L16_float                    (16)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_L16A16_float                 (17)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_R16G16B16A16_float                   (18)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_R11G11B10_float                      (19)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_P128                 (20)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_P32_float                    (21)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_DXT1                 (22)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_DXT1C                        (23)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_DXT3                 (24)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_DXT5                 (25)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_ETC                  (26)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_ETC3                 (27)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_ETC5                 (28)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_LATC1                        (29)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_LATC2                        (30)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_B8G8R8G8                     (31)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_G8B8G8R8                     (32)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_R10G10B10_float_A2                   (33)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_R8G8B8X8                     (34)
#define LWE397_GLOBAL_SURFDESC_0_SURF_FORMAT_B8G8R8X8                     (35)

#define LWE397_GLOBAL_SURFDESC_0_QUADLIN                    1:1
#define LWE397_GLOBAL_SURFDESC_0_QUADLIN_LINEAR                   (0)
#define LWE397_GLOBAL_SURFDESC_0_QUADLIN_TILED                    (1)

#define LWE397_GLOBAL_SURFDESC_0_DITHER                     0:0
#define LWE397_GLOBAL_SURFDESC_0_DITHER_DISABLE                   (0)
#define LWE397_GLOBAL_SURFDESC_0_DITHER_ENABLE                    (1)


// Register LWE397_GLOBAL_SURFDESC  
#define LWE397_GLOBAL_SURFDESC                    (0xe10)
#define LWE397_GLOBAL_SURFDESC_OVERLAP                      27:27
#define LWE397_GLOBAL_SURFDESC_OVERLAP_DISABLE                    (0)
#define LWE397_GLOBAL_SURFDESC_OVERLAP_ENABLE                     (1)

#define LWE397_GLOBAL_SURFDESC_STRUCTURE                    26:25
#define LWE397_GLOBAL_SURFDESC_STRUCTURE_LINEAR                   (0)
#define LWE397_GLOBAL_SURFDESC_STRUCTURE_TILED                    (1)
#define LWE397_GLOBAL_SURFDESC_STRUCTURE_XY_TILED                 (2)

#define LWE397_GLOBAL_SURFDESC_ARRAY_STRIDE                 24:8

#define LWE397_GLOBAL_SURFDESC_STRIDE                       23:8

#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT                  7:2
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_C4X4                   (0)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_A8                     (1)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_L8                     (2)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_S8                     (3)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_L8A8                   (4)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_B2G3R3                 (5)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_B5G6R5                 (6)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_B5G5R5A1                       (7)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_B4G4R4A4                       (8)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_A1B5G5R5                       (9)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_A4B4G4R4                       (10)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_Z16                    (11)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_Z16NL                  (12)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_R8G8B8A8                       (13)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_B8G8R8A8                       (14)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_A16_float                      (15)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_L16_float                      (16)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_L16A16_float                   (17)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_R16G16B16A16_float                     (18)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_R11G11B10_float                        (19)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_P128                   (20)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_P32_float                      (21)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_DXT1                   (22)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_DXT1C                  (23)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_DXT3                   (24)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_DXT5                   (25)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_ETC                    (26)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_ETC3                   (27)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_ETC5                   (28)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_LATC1                  (29)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_LATC2                  (30)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_B8G8R8G8                       (31)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_G8B8G8R8                       (32)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_R10G10B10_float_A2                     (33)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_R8G8B8X8                       (34)
#define LWE397_GLOBAL_SURFDESC_SURF_FORMAT_B8G8R8X8                       (35)

#define LWE397_GLOBAL_SURFDESC_QUADLIN                      1:1
#define LWE397_GLOBAL_SURFDESC_QUADLIN_LINEAR                     (0)
#define LWE397_GLOBAL_SURFDESC_QUADLIN_TILED                      (1)

#define LWE397_GLOBAL_SURFDESC_DITHER                       0:0
#define LWE397_GLOBAL_SURFDESC_DITHER_DISABLE                     (0)
#define LWE397_GLOBAL_SURFDESC_DITHER_ENABLE                      (1)


// Register LWE397_GLOBAL_SURFDESC_1  
#define LWE397_GLOBAL_SURFDESC_1                  (0xe11)
#define LWE397_GLOBAL_SURFDESC_1_OVERLAP                    27:27
#define LWE397_GLOBAL_SURFDESC_1_OVERLAP_DISABLE                  (0)
#define LWE397_GLOBAL_SURFDESC_1_OVERLAP_ENABLE                   (1)

#define LWE397_GLOBAL_SURFDESC_1_STRUCTURE                  26:25
#define LWE397_GLOBAL_SURFDESC_1_STRUCTURE_LINEAR                 (0)
#define LWE397_GLOBAL_SURFDESC_1_STRUCTURE_TILED                  (1)
#define LWE397_GLOBAL_SURFDESC_1_STRUCTURE_XY_TILED                       (2)

#define LWE397_GLOBAL_SURFDESC_1_ARRAY_STRIDE                       24:8

#define LWE397_GLOBAL_SURFDESC_1_STRIDE                     23:8

#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT                        7:2
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_C4X4                 (0)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_A8                   (1)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_L8                   (2)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_S8                   (3)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_L8A8                 (4)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_B2G3R3                       (5)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_B5G6R5                       (6)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_B5G5R5A1                     (7)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_B4G4R4A4                     (8)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_A1B5G5R5                     (9)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_A4B4G4R4                     (10)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_Z16                  (11)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_Z16NL                        (12)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_R8G8B8A8                     (13)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_B8G8R8A8                     (14)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_A16_float                    (15)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_L16_float                    (16)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_L16A16_float                 (17)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_R16G16B16A16_float                   (18)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_R11G11B10_float                      (19)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_P128                 (20)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_P32_float                    (21)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_DXT1                 (22)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_DXT1C                        (23)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_DXT3                 (24)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_DXT5                 (25)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_ETC                  (26)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_ETC3                 (27)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_ETC5                 (28)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_LATC1                        (29)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_LATC2                        (30)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_B8G8R8G8                     (31)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_G8B8G8R8                     (32)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_R10G10B10_float_A2                   (33)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_R8G8B8X8                     (34)
#define LWE397_GLOBAL_SURFDESC_1_SURF_FORMAT_B8G8R8X8                     (35)

#define LWE397_GLOBAL_SURFDESC_1_QUADLIN                    1:1
#define LWE397_GLOBAL_SURFDESC_1_QUADLIN_LINEAR                   (0)
#define LWE397_GLOBAL_SURFDESC_1_QUADLIN_TILED                    (1)

#define LWE397_GLOBAL_SURFDESC_1_DITHER                     0:0
#define LWE397_GLOBAL_SURFDESC_1_DITHER_DISABLE                   (0)
#define LWE397_GLOBAL_SURFDESC_1_DITHER_ENABLE                    (1)


// Register LWE397_GLOBAL_SURFDESC_2  
#define LWE397_GLOBAL_SURFDESC_2                  (0xe12)
#define LWE397_GLOBAL_SURFDESC_2_OVERLAP                    27:27
#define LWE397_GLOBAL_SURFDESC_2_OVERLAP_DISABLE                  (0)
#define LWE397_GLOBAL_SURFDESC_2_OVERLAP_ENABLE                   (1)

#define LWE397_GLOBAL_SURFDESC_2_STRUCTURE                  26:25
#define LWE397_GLOBAL_SURFDESC_2_STRUCTURE_LINEAR                 (0)
#define LWE397_GLOBAL_SURFDESC_2_STRUCTURE_TILED                  (1)
#define LWE397_GLOBAL_SURFDESC_2_STRUCTURE_XY_TILED                       (2)

#define LWE397_GLOBAL_SURFDESC_2_ARRAY_STRIDE                       24:8

#define LWE397_GLOBAL_SURFDESC_2_STRIDE                     23:8

#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT                        7:2
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_C4X4                 (0)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_A8                   (1)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_L8                   (2)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_S8                   (3)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_L8A8                 (4)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_B2G3R3                       (5)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_B5G6R5                       (6)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_B5G5R5A1                     (7)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_B4G4R4A4                     (8)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_A1B5G5R5                     (9)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_A4B4G4R4                     (10)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_Z16                  (11)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_Z16NL                        (12)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_R8G8B8A8                     (13)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_B8G8R8A8                     (14)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_A16_float                    (15)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_L16_float                    (16)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_L16A16_float                 (17)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_R16G16B16A16_float                   (18)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_R11G11B10_float                      (19)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_P128                 (20)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_P32_float                    (21)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_DXT1                 (22)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_DXT1C                        (23)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_DXT3                 (24)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_DXT5                 (25)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_ETC                  (26)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_ETC3                 (27)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_ETC5                 (28)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_LATC1                        (29)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_LATC2                        (30)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_B8G8R8G8                     (31)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_G8B8G8R8                     (32)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_R10G10B10_float_A2                   (33)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_R8G8B8X8                     (34)
#define LWE397_GLOBAL_SURFDESC_2_SURF_FORMAT_B8G8R8X8                     (35)

#define LWE397_GLOBAL_SURFDESC_2_QUADLIN                    1:1
#define LWE397_GLOBAL_SURFDESC_2_QUADLIN_LINEAR                   (0)
#define LWE397_GLOBAL_SURFDESC_2_QUADLIN_TILED                    (1)

#define LWE397_GLOBAL_SURFDESC_2_DITHER                     0:0
#define LWE397_GLOBAL_SURFDESC_2_DITHER_DISABLE                   (0)
#define LWE397_GLOBAL_SURFDESC_2_DITHER_ENABLE                    (1)


// Register LWE397_GLOBAL_SURFDESC_3  
#define LWE397_GLOBAL_SURFDESC_3                  (0xe13)
#define LWE397_GLOBAL_SURFDESC_3_OVERLAP                    27:27
#define LWE397_GLOBAL_SURFDESC_3_OVERLAP_DISABLE                  (0)
#define LWE397_GLOBAL_SURFDESC_3_OVERLAP_ENABLE                   (1)

#define LWE397_GLOBAL_SURFDESC_3_STRUCTURE                  26:25
#define LWE397_GLOBAL_SURFDESC_3_STRUCTURE_LINEAR                 (0)
#define LWE397_GLOBAL_SURFDESC_3_STRUCTURE_TILED                  (1)
#define LWE397_GLOBAL_SURFDESC_3_STRUCTURE_XY_TILED                       (2)

#define LWE397_GLOBAL_SURFDESC_3_ARRAY_STRIDE                       24:8

#define LWE397_GLOBAL_SURFDESC_3_STRIDE                     23:8

#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT                        7:2
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_C4X4                 (0)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_A8                   (1)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_L8                   (2)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_S8                   (3)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_L8A8                 (4)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_B2G3R3                       (5)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_B5G6R5                       (6)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_B5G5R5A1                     (7)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_B4G4R4A4                     (8)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_A1B5G5R5                     (9)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_A4B4G4R4                     (10)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_Z16                  (11)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_Z16NL                        (12)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_R8G8B8A8                     (13)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_B8G8R8A8                     (14)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_A16_float                    (15)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_L16_float                    (16)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_L16A16_float                 (17)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_R16G16B16A16_float                   (18)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_R11G11B10_float                      (19)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_P128                 (20)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_P32_float                    (21)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_DXT1                 (22)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_DXT1C                        (23)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_DXT3                 (24)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_DXT5                 (25)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_ETC                  (26)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_ETC3                 (27)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_ETC5                 (28)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_LATC1                        (29)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_LATC2                        (30)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_B8G8R8G8                     (31)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_G8B8G8R8                     (32)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_R10G10B10_float_A2                   (33)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_R8G8B8X8                     (34)
#define LWE397_GLOBAL_SURFDESC_3_SURF_FORMAT_B8G8R8X8                     (35)

#define LWE397_GLOBAL_SURFDESC_3_QUADLIN                    1:1
#define LWE397_GLOBAL_SURFDESC_3_QUADLIN_LINEAR                   (0)
#define LWE397_GLOBAL_SURFDESC_3_QUADLIN_TILED                    (1)

#define LWE397_GLOBAL_SURFDESC_3_DITHER                     0:0
#define LWE397_GLOBAL_SURFDESC_3_DITHER_DISABLE                   (0)
#define LWE397_GLOBAL_SURFDESC_3_DITHER_ENABLE                    (1)


// Register LWE397_GLOBAL_SURFDESC_4  
#define LWE397_GLOBAL_SURFDESC_4                  (0xe14)
#define LWE397_GLOBAL_SURFDESC_4_OVERLAP                    27:27
#define LWE397_GLOBAL_SURFDESC_4_OVERLAP_DISABLE                  (0)
#define LWE397_GLOBAL_SURFDESC_4_OVERLAP_ENABLE                   (1)

#define LWE397_GLOBAL_SURFDESC_4_STRUCTURE                  26:25
#define LWE397_GLOBAL_SURFDESC_4_STRUCTURE_LINEAR                 (0)
#define LWE397_GLOBAL_SURFDESC_4_STRUCTURE_TILED                  (1)
#define LWE397_GLOBAL_SURFDESC_4_STRUCTURE_XY_TILED                       (2)

#define LWE397_GLOBAL_SURFDESC_4_ARRAY_STRIDE                       24:8

#define LWE397_GLOBAL_SURFDESC_4_STRIDE                     23:8

#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT                        7:2
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_C4X4                 (0)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_A8                   (1)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_L8                   (2)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_S8                   (3)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_L8A8                 (4)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_B2G3R3                       (5)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_B5G6R5                       (6)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_B5G5R5A1                     (7)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_B4G4R4A4                     (8)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_A1B5G5R5                     (9)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_A4B4G4R4                     (10)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_Z16                  (11)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_Z16NL                        (12)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_R8G8B8A8                     (13)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_B8G8R8A8                     (14)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_A16_float                    (15)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_L16_float                    (16)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_L16A16_float                 (17)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_R16G16B16A16_float                   (18)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_R11G11B10_float                      (19)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_P128                 (20)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_P32_float                    (21)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_DXT1                 (22)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_DXT1C                        (23)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_DXT3                 (24)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_DXT5                 (25)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_ETC                  (26)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_ETC3                 (27)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_ETC5                 (28)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_LATC1                        (29)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_LATC2                        (30)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_B8G8R8G8                     (31)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_G8B8G8R8                     (32)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_R10G10B10_float_A2                   (33)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_R8G8B8X8                     (34)
#define LWE397_GLOBAL_SURFDESC_4_SURF_FORMAT_B8G8R8X8                     (35)

#define LWE397_GLOBAL_SURFDESC_4_QUADLIN                    1:1
#define LWE397_GLOBAL_SURFDESC_4_QUADLIN_LINEAR                   (0)
#define LWE397_GLOBAL_SURFDESC_4_QUADLIN_TILED                    (1)

#define LWE397_GLOBAL_SURFDESC_4_DITHER                     0:0
#define LWE397_GLOBAL_SURFDESC_4_DITHER_DISABLE                   (0)
#define LWE397_GLOBAL_SURFDESC_4_DITHER_ENABLE                    (1)


// Register LWE397_GLOBAL_SURFDESC_5  
#define LWE397_GLOBAL_SURFDESC_5                  (0xe15)
#define LWE397_GLOBAL_SURFDESC_5_OVERLAP                    27:27
#define LWE397_GLOBAL_SURFDESC_5_OVERLAP_DISABLE                  (0)
#define LWE397_GLOBAL_SURFDESC_5_OVERLAP_ENABLE                   (1)

#define LWE397_GLOBAL_SURFDESC_5_STRUCTURE                  26:25
#define LWE397_GLOBAL_SURFDESC_5_STRUCTURE_LINEAR                 (0)
#define LWE397_GLOBAL_SURFDESC_5_STRUCTURE_TILED                  (1)
#define LWE397_GLOBAL_SURFDESC_5_STRUCTURE_XY_TILED                       (2)

#define LWE397_GLOBAL_SURFDESC_5_ARRAY_STRIDE                       24:8

#define LWE397_GLOBAL_SURFDESC_5_STRIDE                     23:8

#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT                        7:2
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_C4X4                 (0)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_A8                   (1)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_L8                   (2)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_S8                   (3)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_L8A8                 (4)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_B2G3R3                       (5)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_B5G6R5                       (6)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_B5G5R5A1                     (7)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_B4G4R4A4                     (8)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_A1B5G5R5                     (9)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_A4B4G4R4                     (10)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_Z16                  (11)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_Z16NL                        (12)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_R8G8B8A8                     (13)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_B8G8R8A8                     (14)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_A16_float                    (15)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_L16_float                    (16)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_L16A16_float                 (17)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_R16G16B16A16_float                   (18)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_R11G11B10_float                      (19)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_P128                 (20)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_P32_float                    (21)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_DXT1                 (22)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_DXT1C                        (23)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_DXT3                 (24)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_DXT5                 (25)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_ETC                  (26)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_ETC3                 (27)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_ETC5                 (28)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_LATC1                        (29)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_LATC2                        (30)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_B8G8R8G8                     (31)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_G8B8G8R8                     (32)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_R10G10B10_float_A2                   (33)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_R8G8B8X8                     (34)
#define LWE397_GLOBAL_SURFDESC_5_SURF_FORMAT_B8G8R8X8                     (35)

#define LWE397_GLOBAL_SURFDESC_5_QUADLIN                    1:1
#define LWE397_GLOBAL_SURFDESC_5_QUADLIN_LINEAR                   (0)
#define LWE397_GLOBAL_SURFDESC_5_QUADLIN_TILED                    (1)

#define LWE397_GLOBAL_SURFDESC_5_DITHER                     0:0
#define LWE397_GLOBAL_SURFDESC_5_DITHER_DISABLE                   (0)
#define LWE397_GLOBAL_SURFDESC_5_DITHER_ENABLE                    (1)


// Register LWE397_GLOBAL_SURFDESC_6  
#define LWE397_GLOBAL_SURFDESC_6                  (0xe16)
#define LWE397_GLOBAL_SURFDESC_6_OVERLAP                    27:27
#define LWE397_GLOBAL_SURFDESC_6_OVERLAP_DISABLE                  (0)
#define LWE397_GLOBAL_SURFDESC_6_OVERLAP_ENABLE                   (1)

#define LWE397_GLOBAL_SURFDESC_6_STRUCTURE                  26:25
#define LWE397_GLOBAL_SURFDESC_6_STRUCTURE_LINEAR                 (0)
#define LWE397_GLOBAL_SURFDESC_6_STRUCTURE_TILED                  (1)
#define LWE397_GLOBAL_SURFDESC_6_STRUCTURE_XY_TILED                       (2)

#define LWE397_GLOBAL_SURFDESC_6_ARRAY_STRIDE                       24:8

#define LWE397_GLOBAL_SURFDESC_6_STRIDE                     23:8

#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT                        7:2
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_C4X4                 (0)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_A8                   (1)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_L8                   (2)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_S8                   (3)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_L8A8                 (4)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_B2G3R3                       (5)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_B5G6R5                       (6)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_B5G5R5A1                     (7)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_B4G4R4A4                     (8)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_A1B5G5R5                     (9)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_A4B4G4R4                     (10)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_Z16                  (11)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_Z16NL                        (12)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_R8G8B8A8                     (13)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_B8G8R8A8                     (14)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_A16_float                    (15)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_L16_float                    (16)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_L16A16_float                 (17)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_R16G16B16A16_float                   (18)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_R11G11B10_float                      (19)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_P128                 (20)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_P32_float                    (21)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_DXT1                 (22)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_DXT1C                        (23)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_DXT3                 (24)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_DXT5                 (25)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_ETC                  (26)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_ETC3                 (27)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_ETC5                 (28)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_LATC1                        (29)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_LATC2                        (30)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_B8G8R8G8                     (31)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_G8B8G8R8                     (32)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_R10G10B10_float_A2                   (33)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_R8G8B8X8                     (34)
#define LWE397_GLOBAL_SURFDESC_6_SURF_FORMAT_B8G8R8X8                     (35)

#define LWE397_GLOBAL_SURFDESC_6_QUADLIN                    1:1
#define LWE397_GLOBAL_SURFDESC_6_QUADLIN_LINEAR                   (0)
#define LWE397_GLOBAL_SURFDESC_6_QUADLIN_TILED                    (1)

#define LWE397_GLOBAL_SURFDESC_6_DITHER                     0:0
#define LWE397_GLOBAL_SURFDESC_6_DITHER_DISABLE                   (0)
#define LWE397_GLOBAL_SURFDESC_6_DITHER_ENABLE                    (1)


// Register LWE397_GLOBAL_SURFDESC_7  
#define LWE397_GLOBAL_SURFDESC_7                  (0xe17)
#define LWE397_GLOBAL_SURFDESC_7_OVERLAP                    27:27
#define LWE397_GLOBAL_SURFDESC_7_OVERLAP_DISABLE                  (0)
#define LWE397_GLOBAL_SURFDESC_7_OVERLAP_ENABLE                   (1)

#define LWE397_GLOBAL_SURFDESC_7_STRUCTURE                  26:25
#define LWE397_GLOBAL_SURFDESC_7_STRUCTURE_LINEAR                 (0)
#define LWE397_GLOBAL_SURFDESC_7_STRUCTURE_TILED                  (1)
#define LWE397_GLOBAL_SURFDESC_7_STRUCTURE_XY_TILED                       (2)

#define LWE397_GLOBAL_SURFDESC_7_ARRAY_STRIDE                       24:8

#define LWE397_GLOBAL_SURFDESC_7_STRIDE                     23:8

#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT                        7:2
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_C4X4                 (0)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_A8                   (1)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_L8                   (2)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_S8                   (3)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_L8A8                 (4)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_B2G3R3                       (5)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_B5G6R5                       (6)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_B5G5R5A1                     (7)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_B4G4R4A4                     (8)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_A1B5G5R5                     (9)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_A4B4G4R4                     (10)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_Z16                  (11)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_Z16NL                        (12)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_R8G8B8A8                     (13)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_B8G8R8A8                     (14)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_A16_float                    (15)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_L16_float                    (16)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_L16A16_float                 (17)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_R16G16B16A16_float                   (18)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_R11G11B10_float                      (19)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_P128                 (20)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_P32_float                    (21)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_DXT1                 (22)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_DXT1C                        (23)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_DXT3                 (24)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_DXT5                 (25)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_ETC                  (26)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_ETC3                 (27)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_ETC5                 (28)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_LATC1                        (29)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_LATC2                        (30)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_B8G8R8G8                     (31)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_G8B8G8R8                     (32)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_R10G10B10_float_A2                   (33)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_R8G8B8X8                     (34)
#define LWE397_GLOBAL_SURFDESC_7_SURF_FORMAT_B8G8R8X8                     (35)

#define LWE397_GLOBAL_SURFDESC_7_QUADLIN                    1:1
#define LWE397_GLOBAL_SURFDESC_7_QUADLIN_LINEAR                   (0)
#define LWE397_GLOBAL_SURFDESC_7_QUADLIN_TILED                    (1)

#define LWE397_GLOBAL_SURFDESC_7_DITHER                     0:0
#define LWE397_GLOBAL_SURFDESC_7_DITHER_DISABLE                   (0)
#define LWE397_GLOBAL_SURFDESC_7_DITHER_ENABLE                    (1)


// Register LWE397_GLOBAL_SURFDESC_8  
#define LWE397_GLOBAL_SURFDESC_8                  (0xe18)
#define LWE397_GLOBAL_SURFDESC_8_OVERLAP                    27:27
#define LWE397_GLOBAL_SURFDESC_8_OVERLAP_DISABLE                  (0)
#define LWE397_GLOBAL_SURFDESC_8_OVERLAP_ENABLE                   (1)

#define LWE397_GLOBAL_SURFDESC_8_STRUCTURE                  26:25
#define LWE397_GLOBAL_SURFDESC_8_STRUCTURE_LINEAR                 (0)
#define LWE397_GLOBAL_SURFDESC_8_STRUCTURE_TILED                  (1)
#define LWE397_GLOBAL_SURFDESC_8_STRUCTURE_XY_TILED                       (2)

#define LWE397_GLOBAL_SURFDESC_8_ARRAY_STRIDE                       24:8

#define LWE397_GLOBAL_SURFDESC_8_STRIDE                     23:8

#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT                        7:2
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_C4X4                 (0)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_A8                   (1)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_L8                   (2)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_S8                   (3)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_L8A8                 (4)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_B2G3R3                       (5)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_B5G6R5                       (6)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_B5G5R5A1                     (7)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_B4G4R4A4                     (8)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_A1B5G5R5                     (9)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_A4B4G4R4                     (10)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_Z16                  (11)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_Z16NL                        (12)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_R8G8B8A8                     (13)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_B8G8R8A8                     (14)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_A16_float                    (15)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_L16_float                    (16)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_L16A16_float                 (17)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_R16G16B16A16_float                   (18)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_R11G11B10_float                      (19)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_P128                 (20)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_P32_float                    (21)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_DXT1                 (22)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_DXT1C                        (23)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_DXT3                 (24)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_DXT5                 (25)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_ETC                  (26)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_ETC3                 (27)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_ETC5                 (28)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_LATC1                        (29)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_LATC2                        (30)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_B8G8R8G8                     (31)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_G8B8G8R8                     (32)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_R10G10B10_float_A2                   (33)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_R8G8B8X8                     (34)
#define LWE397_GLOBAL_SURFDESC_8_SURF_FORMAT_B8G8R8X8                     (35)

#define LWE397_GLOBAL_SURFDESC_8_QUADLIN                    1:1
#define LWE397_GLOBAL_SURFDESC_8_QUADLIN_LINEAR                   (0)
#define LWE397_GLOBAL_SURFDESC_8_QUADLIN_TILED                    (1)

#define LWE397_GLOBAL_SURFDESC_8_DITHER                     0:0
#define LWE397_GLOBAL_SURFDESC_8_DITHER_DISABLE                   (0)
#define LWE397_GLOBAL_SURFDESC_8_DITHER_ENABLE                    (1)


// Register LWE397_GLOBAL_SURFDESC_9  
#define LWE397_GLOBAL_SURFDESC_9                  (0xe19)
#define LWE397_GLOBAL_SURFDESC_9_OVERLAP                    27:27
#define LWE397_GLOBAL_SURFDESC_9_OVERLAP_DISABLE                  (0)
#define LWE397_GLOBAL_SURFDESC_9_OVERLAP_ENABLE                   (1)

#define LWE397_GLOBAL_SURFDESC_9_STRUCTURE                  26:25
#define LWE397_GLOBAL_SURFDESC_9_STRUCTURE_LINEAR                 (0)
#define LWE397_GLOBAL_SURFDESC_9_STRUCTURE_TILED                  (1)
#define LWE397_GLOBAL_SURFDESC_9_STRUCTURE_XY_TILED                       (2)

#define LWE397_GLOBAL_SURFDESC_9_ARRAY_STRIDE                       24:8

#define LWE397_GLOBAL_SURFDESC_9_STRIDE                     23:8

#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT                        7:2
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_C4X4                 (0)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_A8                   (1)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_L8                   (2)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_S8                   (3)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_L8A8                 (4)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_B2G3R3                       (5)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_B5G6R5                       (6)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_B5G5R5A1                     (7)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_B4G4R4A4                     (8)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_A1B5G5R5                     (9)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_A4B4G4R4                     (10)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_Z16                  (11)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_Z16NL                        (12)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_R8G8B8A8                     (13)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_B8G8R8A8                     (14)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_A16_float                    (15)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_L16_float                    (16)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_L16A16_float                 (17)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_R16G16B16A16_float                   (18)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_R11G11B10_float                      (19)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_P128                 (20)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_P32_float                    (21)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_DXT1                 (22)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_DXT1C                        (23)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_DXT3                 (24)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_DXT5                 (25)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_ETC                  (26)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_ETC3                 (27)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_ETC5                 (28)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_LATC1                        (29)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_LATC2                        (30)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_B8G8R8G8                     (31)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_G8B8G8R8                     (32)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_R10G10B10_float_A2                   (33)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_R8G8B8X8                     (34)
#define LWE397_GLOBAL_SURFDESC_9_SURF_FORMAT_B8G8R8X8                     (35)

#define LWE397_GLOBAL_SURFDESC_9_QUADLIN                    1:1
#define LWE397_GLOBAL_SURFDESC_9_QUADLIN_LINEAR                   (0)
#define LWE397_GLOBAL_SURFDESC_9_QUADLIN_TILED                    (1)

#define LWE397_GLOBAL_SURFDESC_9_DITHER                     0:0
#define LWE397_GLOBAL_SURFDESC_9_DITHER_DISABLE                   (0)
#define LWE397_GLOBAL_SURFDESC_9_DITHER_ENABLE                    (1)


// Register LWE397_GLOBAL_SURFDESC_10  
#define LWE397_GLOBAL_SURFDESC_10                 (0xe1a)
#define LWE397_GLOBAL_SURFDESC_10_OVERLAP                   27:27
#define LWE397_GLOBAL_SURFDESC_10_OVERLAP_DISABLE                 (0)
#define LWE397_GLOBAL_SURFDESC_10_OVERLAP_ENABLE                  (1)

#define LWE397_GLOBAL_SURFDESC_10_STRUCTURE                 26:25
#define LWE397_GLOBAL_SURFDESC_10_STRUCTURE_LINEAR                        (0)
#define LWE397_GLOBAL_SURFDESC_10_STRUCTURE_TILED                 (1)
#define LWE397_GLOBAL_SURFDESC_10_STRUCTURE_XY_TILED                      (2)

#define LWE397_GLOBAL_SURFDESC_10_ARRAY_STRIDE                      24:8

#define LWE397_GLOBAL_SURFDESC_10_STRIDE                    23:8

#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT                       7:2
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_C4X4                        (0)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_A8                  (1)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_L8                  (2)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_S8                  (3)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_L8A8                        (4)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_B2G3R3                      (5)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_B5G6R5                      (6)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_B5G5R5A1                    (7)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_B4G4R4A4                    (8)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_A1B5G5R5                    (9)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_A4B4G4R4                    (10)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_Z16                 (11)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_Z16NL                       (12)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_R8G8B8A8                    (13)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_B8G8R8A8                    (14)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_A16_float                   (15)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_L16_float                   (16)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_L16A16_float                        (17)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_R16G16B16A16_float                  (18)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_R11G11B10_float                     (19)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_P128                        (20)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_P32_float                   (21)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_DXT1                        (22)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_DXT1C                       (23)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_DXT3                        (24)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_DXT5                        (25)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_ETC                 (26)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_ETC3                        (27)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_ETC5                        (28)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_LATC1                       (29)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_LATC2                       (30)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_B8G8R8G8                    (31)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_G8B8G8R8                    (32)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_R10G10B10_float_A2                  (33)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_R8G8B8X8                    (34)
#define LWE397_GLOBAL_SURFDESC_10_SURF_FORMAT_B8G8R8X8                    (35)

#define LWE397_GLOBAL_SURFDESC_10_QUADLIN                   1:1
#define LWE397_GLOBAL_SURFDESC_10_QUADLIN_LINEAR                  (0)
#define LWE397_GLOBAL_SURFDESC_10_QUADLIN_TILED                   (1)

#define LWE397_GLOBAL_SURFDESC_10_DITHER                    0:0
#define LWE397_GLOBAL_SURFDESC_10_DITHER_DISABLE                  (0)
#define LWE397_GLOBAL_SURFDESC_10_DITHER_ENABLE                   (1)


// Register LWE397_GLOBAL_SURFDESC_11  
#define LWE397_GLOBAL_SURFDESC_11                 (0xe1b)
#define LWE397_GLOBAL_SURFDESC_11_OVERLAP                   27:27
#define LWE397_GLOBAL_SURFDESC_11_OVERLAP_DISABLE                 (0)
#define LWE397_GLOBAL_SURFDESC_11_OVERLAP_ENABLE                  (1)

#define LWE397_GLOBAL_SURFDESC_11_STRUCTURE                 26:25
#define LWE397_GLOBAL_SURFDESC_11_STRUCTURE_LINEAR                        (0)
#define LWE397_GLOBAL_SURFDESC_11_STRUCTURE_TILED                 (1)
#define LWE397_GLOBAL_SURFDESC_11_STRUCTURE_XY_TILED                      (2)

#define LWE397_GLOBAL_SURFDESC_11_ARRAY_STRIDE                      24:8

#define LWE397_GLOBAL_SURFDESC_11_STRIDE                    23:8

#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT                       7:2
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_C4X4                        (0)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_A8                  (1)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_L8                  (2)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_S8                  (3)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_L8A8                        (4)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_B2G3R3                      (5)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_B5G6R5                      (6)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_B5G5R5A1                    (7)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_B4G4R4A4                    (8)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_A1B5G5R5                    (9)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_A4B4G4R4                    (10)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_Z16                 (11)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_Z16NL                       (12)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_R8G8B8A8                    (13)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_B8G8R8A8                    (14)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_A16_float                   (15)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_L16_float                   (16)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_L16A16_float                        (17)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_R16G16B16A16_float                  (18)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_R11G11B10_float                     (19)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_P128                        (20)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_P32_float                   (21)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_DXT1                        (22)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_DXT1C                       (23)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_DXT3                        (24)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_DXT5                        (25)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_ETC                 (26)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_ETC3                        (27)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_ETC5                        (28)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_LATC1                       (29)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_LATC2                       (30)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_B8G8R8G8                    (31)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_G8B8G8R8                    (32)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_R10G10B10_float_A2                  (33)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_R8G8B8X8                    (34)
#define LWE397_GLOBAL_SURFDESC_11_SURF_FORMAT_B8G8R8X8                    (35)

#define LWE397_GLOBAL_SURFDESC_11_QUADLIN                   1:1
#define LWE397_GLOBAL_SURFDESC_11_QUADLIN_LINEAR                  (0)
#define LWE397_GLOBAL_SURFDESC_11_QUADLIN_TILED                   (1)

#define LWE397_GLOBAL_SURFDESC_11_DITHER                    0:0
#define LWE397_GLOBAL_SURFDESC_11_DITHER_DISABLE                  (0)
#define LWE397_GLOBAL_SURFDESC_11_DITHER_ENABLE                   (1)


// Register LWE397_GLOBAL_SURFDESC_12  
#define LWE397_GLOBAL_SURFDESC_12                 (0xe1c)
#define LWE397_GLOBAL_SURFDESC_12_OVERLAP                   27:27
#define LWE397_GLOBAL_SURFDESC_12_OVERLAP_DISABLE                 (0)
#define LWE397_GLOBAL_SURFDESC_12_OVERLAP_ENABLE                  (1)

#define LWE397_GLOBAL_SURFDESC_12_STRUCTURE                 26:25
#define LWE397_GLOBAL_SURFDESC_12_STRUCTURE_LINEAR                        (0)
#define LWE397_GLOBAL_SURFDESC_12_STRUCTURE_TILED                 (1)
#define LWE397_GLOBAL_SURFDESC_12_STRUCTURE_XY_TILED                      (2)

#define LWE397_GLOBAL_SURFDESC_12_ARRAY_STRIDE                      24:8

#define LWE397_GLOBAL_SURFDESC_12_STRIDE                    23:8

#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT                       7:2
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_C4X4                        (0)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_A8                  (1)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_L8                  (2)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_S8                  (3)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_L8A8                        (4)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_B2G3R3                      (5)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_B5G6R5                      (6)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_B5G5R5A1                    (7)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_B4G4R4A4                    (8)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_A1B5G5R5                    (9)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_A4B4G4R4                    (10)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_Z16                 (11)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_Z16NL                       (12)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_R8G8B8A8                    (13)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_B8G8R8A8                    (14)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_A16_float                   (15)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_L16_float                   (16)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_L16A16_float                        (17)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_R16G16B16A16_float                  (18)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_R11G11B10_float                     (19)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_P128                        (20)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_P32_float                   (21)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_DXT1                        (22)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_DXT1C                       (23)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_DXT3                        (24)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_DXT5                        (25)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_ETC                 (26)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_ETC3                        (27)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_ETC5                        (28)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_LATC1                       (29)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_LATC2                       (30)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_B8G8R8G8                    (31)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_G8B8G8R8                    (32)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_R10G10B10_float_A2                  (33)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_R8G8B8X8                    (34)
#define LWE397_GLOBAL_SURFDESC_12_SURF_FORMAT_B8G8R8X8                    (35)

#define LWE397_GLOBAL_SURFDESC_12_QUADLIN                   1:1
#define LWE397_GLOBAL_SURFDESC_12_QUADLIN_LINEAR                  (0)
#define LWE397_GLOBAL_SURFDESC_12_QUADLIN_TILED                   (1)

#define LWE397_GLOBAL_SURFDESC_12_DITHER                    0:0
#define LWE397_GLOBAL_SURFDESC_12_DITHER_DISABLE                  (0)
#define LWE397_GLOBAL_SURFDESC_12_DITHER_ENABLE                   (1)


// Register LWE397_GLOBAL_SURFDESC_13  
#define LWE397_GLOBAL_SURFDESC_13                 (0xe1d)
#define LWE397_GLOBAL_SURFDESC_13_OVERLAP                   27:27
#define LWE397_GLOBAL_SURFDESC_13_OVERLAP_DISABLE                 (0)
#define LWE397_GLOBAL_SURFDESC_13_OVERLAP_ENABLE                  (1)

#define LWE397_GLOBAL_SURFDESC_13_STRUCTURE                 26:25
#define LWE397_GLOBAL_SURFDESC_13_STRUCTURE_LINEAR                        (0)
#define LWE397_GLOBAL_SURFDESC_13_STRUCTURE_TILED                 (1)
#define LWE397_GLOBAL_SURFDESC_13_STRUCTURE_XY_TILED                      (2)

#define LWE397_GLOBAL_SURFDESC_13_ARRAY_STRIDE                      24:8

#define LWE397_GLOBAL_SURFDESC_13_STRIDE                    23:8

#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT                       7:2
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_C4X4                        (0)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_A8                  (1)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_L8                  (2)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_S8                  (3)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_L8A8                        (4)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_B2G3R3                      (5)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_B5G6R5                      (6)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_B5G5R5A1                    (7)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_B4G4R4A4                    (8)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_A1B5G5R5                    (9)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_A4B4G4R4                    (10)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_Z16                 (11)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_Z16NL                       (12)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_R8G8B8A8                    (13)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_B8G8R8A8                    (14)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_A16_float                   (15)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_L16_float                   (16)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_L16A16_float                        (17)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_R16G16B16A16_float                  (18)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_R11G11B10_float                     (19)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_P128                        (20)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_P32_float                   (21)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_DXT1                        (22)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_DXT1C                       (23)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_DXT3                        (24)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_DXT5                        (25)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_ETC                 (26)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_ETC3                        (27)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_ETC5                        (28)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_LATC1                       (29)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_LATC2                       (30)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_B8G8R8G8                    (31)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_G8B8G8R8                    (32)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_R10G10B10_float_A2                  (33)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_R8G8B8X8                    (34)
#define LWE397_GLOBAL_SURFDESC_13_SURF_FORMAT_B8G8R8X8                    (35)

#define LWE397_GLOBAL_SURFDESC_13_QUADLIN                   1:1
#define LWE397_GLOBAL_SURFDESC_13_QUADLIN_LINEAR                  (0)
#define LWE397_GLOBAL_SURFDESC_13_QUADLIN_TILED                   (1)

#define LWE397_GLOBAL_SURFDESC_13_DITHER                    0:0
#define LWE397_GLOBAL_SURFDESC_13_DITHER_DISABLE                  (0)
#define LWE397_GLOBAL_SURFDESC_13_DITHER_ENABLE                   (1)


// Register LWE397_GLOBAL_SURFDESC_14  
#define LWE397_GLOBAL_SURFDESC_14                 (0xe1e)
#define LWE397_GLOBAL_SURFDESC_14_OVERLAP                   27:27
#define LWE397_GLOBAL_SURFDESC_14_OVERLAP_DISABLE                 (0)
#define LWE397_GLOBAL_SURFDESC_14_OVERLAP_ENABLE                  (1)

#define LWE397_GLOBAL_SURFDESC_14_STRUCTURE                 26:25
#define LWE397_GLOBAL_SURFDESC_14_STRUCTURE_LINEAR                        (0)
#define LWE397_GLOBAL_SURFDESC_14_STRUCTURE_TILED                 (1)
#define LWE397_GLOBAL_SURFDESC_14_STRUCTURE_XY_TILED                      (2)

#define LWE397_GLOBAL_SURFDESC_14_ARRAY_STRIDE                      24:8

#define LWE397_GLOBAL_SURFDESC_14_STRIDE                    23:8

#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT                       7:2
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_C4X4                        (0)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_A8                  (1)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_L8                  (2)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_S8                  (3)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_L8A8                        (4)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_B2G3R3                      (5)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_B5G6R5                      (6)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_B5G5R5A1                    (7)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_B4G4R4A4                    (8)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_A1B5G5R5                    (9)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_A4B4G4R4                    (10)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_Z16                 (11)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_Z16NL                       (12)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_R8G8B8A8                    (13)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_B8G8R8A8                    (14)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_A16_float                   (15)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_L16_float                   (16)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_L16A16_float                        (17)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_R16G16B16A16_float                  (18)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_R11G11B10_float                     (19)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_P128                        (20)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_P32_float                   (21)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_DXT1                        (22)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_DXT1C                       (23)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_DXT3                        (24)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_DXT5                        (25)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_ETC                 (26)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_ETC3                        (27)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_ETC5                        (28)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_LATC1                       (29)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_LATC2                       (30)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_B8G8R8G8                    (31)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_G8B8G8R8                    (32)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_R10G10B10_float_A2                  (33)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_R8G8B8X8                    (34)
#define LWE397_GLOBAL_SURFDESC_14_SURF_FORMAT_B8G8R8X8                    (35)

#define LWE397_GLOBAL_SURFDESC_14_QUADLIN                   1:1
#define LWE397_GLOBAL_SURFDESC_14_QUADLIN_LINEAR                  (0)
#define LWE397_GLOBAL_SURFDESC_14_QUADLIN_TILED                   (1)

#define LWE397_GLOBAL_SURFDESC_14_DITHER                    0:0
#define LWE397_GLOBAL_SURFDESC_14_DITHER_DISABLE                  (0)
#define LWE397_GLOBAL_SURFDESC_14_DITHER_ENABLE                   (1)


// Register LWE397_GLOBAL_SURFDESC_15  
#define LWE397_GLOBAL_SURFDESC_15                 (0xe1f)
#define LWE397_GLOBAL_SURFDESC_15_OVERLAP                   27:27
#define LWE397_GLOBAL_SURFDESC_15_OVERLAP_DISABLE                 (0)
#define LWE397_GLOBAL_SURFDESC_15_OVERLAP_ENABLE                  (1)

#define LWE397_GLOBAL_SURFDESC_15_STRUCTURE                 26:25
#define LWE397_GLOBAL_SURFDESC_15_STRUCTURE_LINEAR                        (0)
#define LWE397_GLOBAL_SURFDESC_15_STRUCTURE_TILED                 (1)
#define LWE397_GLOBAL_SURFDESC_15_STRUCTURE_XY_TILED                      (2)

#define LWE397_GLOBAL_SURFDESC_15_ARRAY_STRIDE                      24:8

#define LWE397_GLOBAL_SURFDESC_15_STRIDE                    23:8

#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT                       7:2
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_C4X4                        (0)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_A8                  (1)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_L8                  (2)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_S8                  (3)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_L8A8                        (4)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_B2G3R3                      (5)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_B5G6R5                      (6)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_B5G5R5A1                    (7)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_B4G4R4A4                    (8)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_A1B5G5R5                    (9)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_A4B4G4R4                    (10)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_Z16                 (11)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_Z16NL                       (12)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_R8G8B8A8                    (13)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_B8G8R8A8                    (14)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_A16_float                   (15)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_L16_float                   (16)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_L16A16_float                        (17)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_R16G16B16A16_float                  (18)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_R11G11B10_float                     (19)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_P128                        (20)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_P32_float                   (21)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_DXT1                        (22)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_DXT1C                       (23)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_DXT3                        (24)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_DXT5                        (25)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_ETC                 (26)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_ETC3                        (27)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_ETC5                        (28)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_LATC1                       (29)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_LATC2                       (30)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_B8G8R8G8                    (31)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_G8B8G8R8                    (32)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_R10G10B10_float_A2                  (33)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_R8G8B8X8                    (34)
#define LWE397_GLOBAL_SURFDESC_15_SURF_FORMAT_B8G8R8X8                    (35)

#define LWE397_GLOBAL_SURFDESC_15_QUADLIN                   1:1
#define LWE397_GLOBAL_SURFDESC_15_QUADLIN_LINEAR                  (0)
#define LWE397_GLOBAL_SURFDESC_15_QUADLIN_TILED                   (1)

#define LWE397_GLOBAL_SURFDESC_15_DITHER                    0:0
#define LWE397_GLOBAL_SURFDESC_15_DITHER_DISABLE                  (0)
#define LWE397_GLOBAL_SURFDESC_15_DITHER_ENABLE                   (1)


// Register LWE397_GLOBAL_PIX_ATTR_0  
#define LWE397_GLOBAL_PIX_ATTR_0                  (0xe20)
#define LWE397_GLOBAL_PIX_ATTR_0_NUM_ROWS                   1:0

#define LWE397_GLOBAL_PIX_ATTR_0_FIRST_SEQUENCE                     13:8

#define LWE397_GLOBAL_PIX_ATTR_0_MAX_QID                    31:24


// Register LWE397_GLOBAL_TRI_ATTR_0  
#define LWE397_GLOBAL_TRI_ATTR_0                  (0xe21)
#define LWE397_GLOBAL_TRI_ATTR_0_NUM_TRIS                   6:0

#define LWE397_GLOBAL_TRI_ATTR_0_TRI_ROWS                   14:8


// Register LWE397_GLOBAL_INST_OFFSET_0  
#define LWE397_GLOBAL_INST_OFFSET_0                       (0xe22)
#define LWE397_GLOBAL_INST_OFFSET_0_INDEX                   5:0


// Register LWE397_GLOBAL_RAISE_0  
#define LWE397_GLOBAL_RAISE_0                     (0xe23)
#define LWE397_GLOBAL_RAISE_0_SIGNAL                        4:0

#define LWE397_GLOBAL_RAISE_0_CHANNEL                       11:8

#define LWE397_GLOBAL_RAISE_0_SYNCPT_INCR                   31:31
#define LWE397_GLOBAL_RAISE_0_SYNCPT_INCR_DISABLE                 (0)
#define LWE397_GLOBAL_RAISE_0_SYNCPT_INCR_ENABLE                  (1)


// Register LWE397_GLOBAL_REFCNT_0  
#define LWE397_GLOBAL_REFCNT_0                    (0xe24)
#define LWE397_GLOBAL_REFCNT_0_VALUE                        31:0

#define LWE397_GLOBAL_REFCNT_0_SYNCPT_INCR                  31:31
#define LWE397_GLOBAL_REFCNT_0_SYNCPT_INCR_DISABLE                        (0)
#define LWE397_GLOBAL_REFCNT_0_SYNCPT_INCR_ENABLE                 (1)


// Register LWE397_GLOBAL_INSTRUMENT_0  
#define LWE397_GLOBAL_INSTRUMENT_0                        (0xe25)
#define LWE397_GLOBAL_INSTRUMENT_0_STAT_EN                  0:0


// Register LWE397_GLOBAL_DITHER_TABLE_0  
#define LWE397_GLOBAL_DITHER_TABLE_0                      (0xe26)
#define LWE397_GLOBAL_DITHER_TABLE_0_DITHER_ALPHA                   12:12
#define LWE397_GLOBAL_DITHER_TABLE_0_DITHER_ALPHA_DISABLE                 (0)
#define LWE397_GLOBAL_DITHER_TABLE_0_DITHER_ALPHA_ENABLE                  (1)

#define LWE397_GLOBAL_DITHER_TABLE_0_ONE_ONE                        11:9

#define LWE397_GLOBAL_DITHER_TABLE_0_ONE_ZERO                       8:6

#define LWE397_GLOBAL_DITHER_TABLE_0_ZERO_ONE                       5:3

#define LWE397_GLOBAL_DITHER_TABLE_0_ZERO_ZERO                      2:0


// Register LWE397_GLOBAL_FLUSH_0  
#define LWE397_GLOBAL_FLUSH_0                     (0xe27)
#define LWE397_GLOBAL_FLUSH_0_TEXTURE                       0:0

#define LWE397_GLOBAL_FLUSH_0_QRAST                 1:1

#define LWE397_GLOBAL_FLUSH_0_GR3D                  2:2

// Register LWE397_GLOBAL_S_OPERATION_0  
#define LWE397_GLOBAL_S_OPERATION_0                       (0xe28)
#define LWE397_GLOBAL_S_OPERATION_0_S_FAIL                  2:0

#define LWE397_GLOBAL_S_OPERATION_0_Z_FAIL                  5:3

#define LWE397_GLOBAL_S_OPERATION_0_Z_PASS                  8:6

#define LWE397_GLOBAL_S_OPERATION_0_S_WR_MASK                       16:9

#define LWE397_GLOBAL_S_OPERATION_0_S_REF                   24:17


// Register LWE397_GLOBAL_S_OPERATION  
#define LWE397_GLOBAL_S_OPERATION                 (0xe28)
#define LWE397_GLOBAL_S_OPERATION_S_FAIL                    2:0

#define LWE397_GLOBAL_S_OPERATION_Z_FAIL                    5:3

#define LWE397_GLOBAL_S_OPERATION_Z_PASS                    8:6

#define LWE397_GLOBAL_S_OPERATION_S_WR_MASK                 16:9

#define LWE397_GLOBAL_S_OPERATION_S_REF                     24:17


// Register LWE397_GLOBAL_S_OPERATION_1  
#define LWE397_GLOBAL_S_OPERATION_1                       (0xe29)
#define LWE397_GLOBAL_S_OPERATION_1_S_FAIL                  2:0

#define LWE397_GLOBAL_S_OPERATION_1_Z_FAIL                  5:3

#define LWE397_GLOBAL_S_OPERATION_1_Z_PASS                  8:6

#define LWE397_GLOBAL_S_OPERATION_1_S_WR_MASK                       16:9

#define LWE397_GLOBAL_S_OPERATION_1_S_REF                   24:17


// Register LWE397_GLOBAL_SPILLSURFADDR_0  
#define LWE397_GLOBAL_SPILLSURFADDR_0                     (0xe2a)
#define LWE397_GLOBAL_SPILLSURFADDR_0_BASE_ADDRESS                  31:0


// Register LWE397_GLOBAL_LW_MCCIF_FIFOCTRL_0  
#define LWE397_GLOBAL_LW_MCCIF_FIFOCTRL_0                 (0xe2b)
#define LWE397_GLOBAL_LW_MCCIF_FIFOCTRL_0_LW_MCCIF_WRCL_MCLE2X                      0:0
#define LWE397_GLOBAL_LW_MCCIF_FIFOCTRL_0_LW_MCCIF_WRCL_MCLE2X_DISABLE                    (0)
#define LWE397_GLOBAL_LW_MCCIF_FIFOCTRL_0_LW_MCCIF_WRCL_MCLE2X_ENABLE                     (1)

#define LWE397_GLOBAL_LW_MCCIF_FIFOCTRL_0_LW_MCCIF_RDMC_RDFAST                      1:1
#define LWE397_GLOBAL_LW_MCCIF_FIFOCTRL_0_LW_MCCIF_RDMC_RDFAST_DISABLE                    (0)
#define LWE397_GLOBAL_LW_MCCIF_FIFOCTRL_0_LW_MCCIF_RDMC_RDFAST_ENABLE                     (1)

#define LWE397_GLOBAL_LW_MCCIF_FIFOCTRL_0_LW_MCCIF_WRMC_CLLE2X                      2:2
#define LWE397_GLOBAL_LW_MCCIF_FIFOCTRL_0_LW_MCCIF_WRMC_CLLE2X_DISABLE                    (0)
#define LWE397_GLOBAL_LW_MCCIF_FIFOCTRL_0_LW_MCCIF_WRMC_CLLE2X_ENABLE                     (1)

#define LWE397_GLOBAL_LW_MCCIF_FIFOCTRL_0_LW_MCCIF_RDCL_RDFAST                      3:3
#define LWE397_GLOBAL_LW_MCCIF_FIFOCTRL_0_LW_MCCIF_RDCL_RDFAST_DISABLE                    (0)
#define LWE397_GLOBAL_LW_MCCIF_FIFOCTRL_0_LW_MCCIF_RDCL_RDFAST_ENABLE                     (1)

#define LWE397_GLOBAL_LW_MCCIF_FIFOCTRL_0_LW_WCLK_OVERRIDE                  16:16

#define LWE397_GLOBAL_LW_MCCIF_FIFOCTRL_0_LW_RCLK_OVERRIDE                  17:17


// Register LWE397_GLOBAL_SURFOVERADDR_0  
#define LWE397_GLOBAL_SURFOVERADDR_0                      (0xe30)
#define LWE397_GLOBAL_SURFOVERADDR_0_BASE_ADDRESS                   31:0


// Register LWE397_GLOBAL_SURFOVERADDR  
#define LWE397_GLOBAL_SURFOVERADDR                        (0xe30)
#define LWE397_GLOBAL_SURFOVERADDR_BASE_ADDRESS                     31:0


// Register LWE397_GLOBAL_SURFOVERADDR_1  
#define LWE397_GLOBAL_SURFOVERADDR_1                      (0xe31)
#define LWE397_GLOBAL_SURFOVERADDR_1_BASE_ADDRESS                   31:0


// Register LWE397_GLOBAL_SURFOVERADDR_2  
#define LWE397_GLOBAL_SURFOVERADDR_2                      (0xe32)
#define LWE397_GLOBAL_SURFOVERADDR_2_BASE_ADDRESS                   31:0


// Register LWE397_GLOBAL_SURFOVERADDR_3  
#define LWE397_GLOBAL_SURFOVERADDR_3                      (0xe33)
#define LWE397_GLOBAL_SURFOVERADDR_3_BASE_ADDRESS                   31:0


// Register LWE397_GLOBAL_SURFOVERADDR_4  
#define LWE397_GLOBAL_SURFOVERADDR_4                      (0xe34)
#define LWE397_GLOBAL_SURFOVERADDR_4_BASE_ADDRESS                   31:0


// Register LWE397_GLOBAL_SURFOVERADDR_5  
#define LWE397_GLOBAL_SURFOVERADDR_5                      (0xe35)
#define LWE397_GLOBAL_SURFOVERADDR_5_BASE_ADDRESS                   31:0


// Register LWE397_GLOBAL_SURFOVERADDR_6  
#define LWE397_GLOBAL_SURFOVERADDR_6                      (0xe36)
#define LWE397_GLOBAL_SURFOVERADDR_6_BASE_ADDRESS                   31:0


// Register LWE397_GLOBAL_SURFOVERADDR_7  
#define LWE397_GLOBAL_SURFOVERADDR_7                      (0xe37)
#define LWE397_GLOBAL_SURFOVERADDR_7_BASE_ADDRESS                   31:0


// Register LWE397_GLOBAL_SURFOVERADDR_8  
#define LWE397_GLOBAL_SURFOVERADDR_8                      (0xe38)
#define LWE397_GLOBAL_SURFOVERADDR_8_BASE_ADDRESS                   31:0


// Register LWE397_GLOBAL_SURFOVERADDR_9  
#define LWE397_GLOBAL_SURFOVERADDR_9                      (0xe39)
#define LWE397_GLOBAL_SURFOVERADDR_9_BASE_ADDRESS                   31:0


// Register LWE397_GLOBAL_SURFOVERADDR_10  
#define LWE397_GLOBAL_SURFOVERADDR_10                     (0xe3a)
#define LWE397_GLOBAL_SURFOVERADDR_10_BASE_ADDRESS                  31:0


// Register LWE397_GLOBAL_SURFOVERADDR_11  
#define LWE397_GLOBAL_SURFOVERADDR_11                     (0xe3b)
#define LWE397_GLOBAL_SURFOVERADDR_11_BASE_ADDRESS                  31:0


// Register LWE397_GLOBAL_SURFOVERADDR_12  
#define LWE397_GLOBAL_SURFOVERADDR_12                     (0xe3c)
#define LWE397_GLOBAL_SURFOVERADDR_12_BASE_ADDRESS                  31:0


// Register LWE397_GLOBAL_SURFOVERADDR_13  
#define LWE397_GLOBAL_SURFOVERADDR_13                     (0xe3d)
#define LWE397_GLOBAL_SURFOVERADDR_13_BASE_ADDRESS                  31:0


// Register LWE397_GLOBAL_SURFOVERADDR_14  
#define LWE397_GLOBAL_SURFOVERADDR_14                     (0xe3e)
#define LWE397_GLOBAL_SURFOVERADDR_14_BASE_ADDRESS                  31:0


// Register LWE397_GLOBAL_SURFOVERADDR_15  
#define LWE397_GLOBAL_SURFOVERADDR_15                     (0xe3f)
#define LWE397_GLOBAL_SURFOVERADDR_15_BASE_ADDRESS                  31:0


// Register LWE397_GLOBAL_MEMORY_OUTPUT_READS_0  
#define LWE397_GLOBAL_MEMORY_OUTPUT_READS_0                       (0xe40)
#define LWE397_GLOBAL_MEMORY_OUTPUT_READS_0_READ_DEST                       0:0
#define LWE397_GLOBAL_MEMORY_OUTPUT_READS_0_READ_DEST_HOST1X                      (0)
#define LWE397_GLOBAL_MEMORY_OUTPUT_READS_0_READ_DEST_MEMORY                      (1)


// Register LWE397_GLOBAL_HORIZONTAL_SWATH_RENDERING_0  
#define LWE397_GLOBAL_HORIZONTAL_SWATH_RENDERING_0                        (0xe41)
#define LWE397_GLOBAL_HORIZONTAL_SWATH_RENDERING_0_ALTERNATE                        0:0
#define LWE397_GLOBAL_HORIZONTAL_SWATH_RENDERING_0_ALTERNATE_DISABLE                      (0)
#define LWE397_GLOBAL_HORIZONTAL_SWATH_RENDERING_0_ALTERNATE_ENABLE                       (1)

#define LWE397_GLOBAL_HORIZONTAL_SWATH_RENDERING_0_SWATH_ALIGN                      1:1
#define LWE397_GLOBAL_HORIZONTAL_SWATH_RENDERING_0_SWATH_ALIGN_EVEN                       (0)
#define LWE397_GLOBAL_HORIZONTAL_SWATH_RENDERING_0_SWATH_ALIGN_ODD                        (1)


// Register LWE397_GLOBAL_INNER_SLI_SCISSOR_X_0  
#define LWE397_GLOBAL_INNER_SLI_SCISSOR_X_0                       (0xe42)
#define LWE397_GLOBAL_INNER_SLI_SCISSOR_X_0_MAX                     12:0

#define LWE397_GLOBAL_INNER_SLI_SCISSOR_X_0_MIN                     27:16


// Register LWE397_GLOBAL_INNER_SLI_SCISSOR_Y_0  
#define LWE397_GLOBAL_INNER_SLI_SCISSOR_Y_0                       (0xe43)
#define LWE397_GLOBAL_INNER_SLI_SCISSOR_Y_0_MAX                     12:0

#define LWE397_GLOBAL_INNER_SLI_SCISSOR_Y_0_MIN                     27:16

/*

#define AR_SNAP_BITS    4
#define SETUP_PACKET_WIDTH      96
#define SETUP_SB_WIDTH  11

// Packet TD_FLOAT
#define TD_FLOAT_SIZE 20

#define TD_FLOAT_FLOAT_ROW                      0

#define TD_FLOAT_MANT_ROW                       0

#define TD_FLOAT_EXP_ROW                        0

#define TD_FLOAT_SIGN_ROW                       0


// Packet IEEE_FLOAT
#define IEEE_FLOAT_SIZE 32

#define IEEE_FLOAT_FLOAT_ROW                    0

#define IEEE_FLOAT_MANT_ROW                     0

#define IEEE_FLOAT_EXP_ROW                      0

#define IEEE_FLOAT_SIGN_ROW                     0


// Packet FP22_FLOAT
#define FP22_FLOAT_SIZE 22

#define FP22_FLOAT_FLOAT_ROW                    0

#define FP22_FLOAT_MANT_ROW                     0
 
#define FP22_FLOAT_EXP_ROW                      0

#define FP22_FLOAT_SIGN_ROW                     0


// Packet FP16_FLOAT
#define FP16_FLOAT_SIZE 16

#define FP16_FLOAT_FLOAT_ROW                    0

#define FP16_FLOAT_MANT_ROW                     0

#define FP16_FLOAT_EXP_ROW                      0

#define FP16_FLOAT_SIGN_ROW                     0


// Packet FP11_FLOAT
#define FP11_FLOAT_SIZE 11

#define FP11_FLOAT_FLOAT_ROW                    0

#define FP11_FLOAT_MANT_ROW                     0

#define FP11_FLOAT_EXP_ROW                      0


// Packet FP10_FLOAT
#define FP10_FLOAT_SIZE 10

#define FP10_FLOAT_FLOAT_ROW                    0

#define FP10_FLOAT_MANT_ROW                     0

#define FP10_FLOAT_EXP_ROW                      0


// Packet FP10_6E4_FLOAT
#define FP10_6E4_FLOAT_SIZE 10

#define FP10_6E4_FLOAT_FLOAT_ROW                        0

#define FP10_6E4_FLOAT_MANT_ROW                 0

#define FP10_6E4_FLOAT_EXP_ROW                  0


// Packet TD_FIX
#define TD_FIX_SIZE 10

#define TD_FIX_FIX_ROW                  0

#define TD_FIX_FRAC_ROW                 0

#define TD_FIX_INT_ROW                  0

#define TD_FIX_SIGN_ROW                 0

#define TD_FIX_ZERO     0
#define TD_FIX_ZERO_SIGN        0
#define TD_FIX_ZERO_INT 0
#define TD_FIX_ZERO_FRAC        0
#define TD_FIX_ONE      256
#define TD_FIX_ONE_SIGN 0
#define TD_FIX_ONE_INT  1
#define TD_FIX_ONE_FRAC 0
#define TD_FIX_MINUS_ONE        768
#define TD_FIX_MINUS_ONE_SIGN   1
#define TD_FIX_MINUS_ONE_INT    1
#define TD_FIX_MINUS_ONE_FRAC   0
#define TD_FLOAT_ZERO   0
#define TD_FLOAT_ZERO_SIGN      0
#define TD_FLOAT_ZERO_MANT      0
#define TD_FLOAT_ZERO_EXP       0
#define TD_FLOAT_ONE    253952
#define TD_FLOAT_ONE_SIGN       0
#define TD_FLOAT_ONE_MANT       0
#define TD_FLOAT_ONE_EXP        31
#define TD_FLOAT_MINUS_ONE      778240
#define TD_FLOAT_MINUS_ONE_SIGN 1
#define TD_FLOAT_MINUS_ONE_MANT 0
#define TD_FLOAT_MINUS_ONE_EXP  31
#define TD_FLOAT_MINUS_MAX      1048575
#define TD_FLOAT_MINUS_MAX_SIGN 1
#define TD_FLOAT_MINUS_MAX_MANT 8191
#define TD_FLOAT_MINUS_MAX_EXP  63
#define TD_FLOAT_PLUS_MAX       524287
#define TD_FLOAT_PLUS_MAX_SIGN  0
#define TD_FLOAT_PLUS_MAX_MANT  8191
#define TD_FLOAT_PLUS_MAX_EXP   63
#define IEEE_FLOAT_PLUS_ZERO    0
#define IEEE_FLOAT_PLUS_ZERO_SIGN       0
#define IEEE_FLOAT_PLUS_ZERO_MANT       0
#define IEEE_FLOAT_PLUS_ZERO_EXP        0
#define IEEE_FLOAT_MINUS_ZERO   -2147483648
#define IEEE_FLOAT_MINUS_ZERO_SIGN      1
#define IEEE_FLOAT_MINUS_ZERO_MANT      0
#define IEEE_FLOAT_MINUS_ZERO_EXP       0
#define IEEE_FLOAT_PLUS_INF     2139095040
#define IEEE_FLOAT_PLUS_INF_SIGN        0
#define IEEE_FLOAT_PLUS_INF_MANT        0
#define IEEE_FLOAT_PLUS_INF_EXP 255
#define IEEE_FLOAT_MINUS_INF    -8388608
#define IEEE_FLOAT_MINUS_INF_SIGN       1
#define IEEE_FLOAT_MINUS_INF_MANT       0
#define IEEE_FLOAT_MINUS_INF_EXP        255
#define IEEE_FLOAT_ONE  1065353216
#define IEEE_FLOAT_ONE_SIGN     0
#define IEEE_FLOAT_ONE_MANT     0
#define IEEE_FLOAT_ONE_EXP      127
#define IEEE_FLOAT_MINUS_ONE    -1082130432
#define IEEE_FLOAT_MINUS_ONE_SIGN       1
#define IEEE_FLOAT_MINUS_ONE_MANT       0
#define IEEE_FLOAT_MINUS_ONE_EXP        127
#define FP22_FLOAT_ZERO 0
#define FP22_FLOAT_ZERO_SIGN    0
#define FP22_FLOAT_ZERO_MANT    0
#define FP22_FLOAT_ZERO_EXP     0
#define FP22_FLOAT_ONE  1015808
#define FP22_FLOAT_ONE_SIGN     0
#define FP22_FLOAT_ONE_MANT     0
#define FP22_FLOAT_ONE_EXP      31
#define FP22_FLOAT_MINUS_ONE    3112960
#define FP22_FLOAT_MINUS_ONE_SIGN       1
#define FP22_FLOAT_MINUS_ONE_MANT       0
#define FP22_FLOAT_MINUS_ONE_EXP        31
#define FP22_FLOAT_MINUS_MAX    4194303
#define FP22_FLOAT_MINUS_MAX_SIGN       1
#define FP22_FLOAT_MINUS_MAX_MANT       16383
#define FP22_FLOAT_MINUS_MAX_EXP        63
#define FP22_FLOAT_PLUS_MAX     2097151
#define FP22_FLOAT_PLUS_MAX_SIGN        0
#define FP22_FLOAT_PLUS_MAX_MANT        16383
#define FP22_FLOAT_PLUS_MAX_EXP 63

// Packet CLIP2SETUP_COV
#define CLIP2SETUP_COV_SIZE 112

#define CLIP2SETUP_COV_TAG_ROW                  0
#define CLIP2SETUP_COV_TAG_REG                  (0)
#define CLIP2SETUP_COV_TAG_PRIM                 (1)
#define CLIP2SETUP_COV_TAG_XY                   (2)
#define CLIP2SETUP_COV_TAG_RHW                  (3)

#define CLIP2SETUP_COV_DATA_ROW                 0


// Packet CLIP2SETUP_COV_REG
#define CLIP2SETUP_COV_REG_SIZE 56

#define CLIP2SETUP_COV_REG_TAG_ROW                      0
#define CLIP2SETUP_COV_REG_TAG_REG                      (0)
#define CLIP2SETUP_COV_REG_TAG_PRIM                     (1)
#define CLIP2SETUP_COV_REG_TAG_XY                       (2)
#define CLIP2SETUP_COV_REG_TAG_RHW                      (3)

#define CLIP2SETUP_COV_REG_READ_ROW                     0

#define CLIP2SETUP_COV_REG_ADDR_ROW                     0

#define CLIP2SETUP_COV_REG_DATA_ROW                     0

#define CLIP2SETUP_COV_REG_CHANNEL_ROW                  0


// Packet CLIP2SETUP_COV_PRIM
#define CLIP2SETUP_COV_PRIM_SIZE 106

#define CLIP2SETUP_COV_PRIM_TAG_ROW                     0
#define CLIP2SETUP_COV_PRIM_TAG_REG                     (0)
#define CLIP2SETUP_COV_PRIM_TAG_PRIM                    (1)
#define CLIP2SETUP_COV_PRIM_TAG_XY                      (2)
#define CLIP2SETUP_COV_PRIM_TAG_RHW                     (3)

#define CLIP2SETUP_COV_PRIM_PRIM_TYPE_ROW                       0
#define CLIP2SETUP_COV_PRIM_PRIM_TYPE_POINT                     (1)
#define CLIP2SETUP_COV_PRIM_PRIM_TYPE_LINE                      (2)
#define CLIP2SETUP_COV_PRIM_PRIM_TYPE_TRI                       (3)

#define CLIP2SETUP_COV_PRIM_TID_ROW                     0

#define CLIP2SETUP_COV_PRIM_AREA_ROW                    0

#define CLIP2SETUP_COV_PRIM_X3_ROW                      0

#define CLIP2SETUP_COV_PRIM_Y3_ROW                      0


// Packet CLIP2SETUP_COV_XY
#define CLIP2SETUP_COV_XY_SIZE 112

#define CLIP2SETUP_COV_XY_TAG_ROW                       0
#define CLIP2SETUP_COV_XY_TAG_REG                       (0)
#define CLIP2SETUP_COV_XY_TAG_PRIM                      (1)
#define CLIP2SETUP_COV_XY_TAG_XY                        (2)
#define CLIP2SETUP_COV_XY_TAG_RHW                       (3)

#define CLIP2SETUP_COV_XY_FACE_ROW                      0
#define CLIP2SETUP_COV_XY_FACE_FRONT                    (0)
#define CLIP2SETUP_COV_XY_FACE_BACK                     (1)

#define CLIP2SETUP_COV_XY_X0_ROW                        0

#define CLIP2SETUP_COV_XY_Y0_ROW                        0

#define CLIP2SETUP_COV_XY_X1_ROW                        0

#define CLIP2SETUP_COV_XY_Y1_ROW                        0

#define CLIP2SETUP_COV_XY_X2_ROW                        0

#define CLIP2SETUP_COV_XY_Y2_ROW                        0


// Packet CLIP2SETUP_COV_RHW
#define CLIP2SETUP_COV_RHW_SIZE 100

#define CLIP2SETUP_COV_RHW_TAG_ROW                      0
#define CLIP2SETUP_COV_RHW_TAG_REG                      (0)
#define CLIP2SETUP_COV_RHW_TAG_PRIM                     (1)
#define CLIP2SETUP_COV_RHW_TAG_XY                       (2)
#define CLIP2SETUP_COV_RHW_TAG_RHW                      (3)

#define CLIP2SETUP_COV_RHW_RHW0_ROW                     0

#define CLIP2SETUP_COV_RHW_RHW1_ROW                     0

#define CLIP2SETUP_COV_RHW_RHW2_ROW                     0


// Packet CLIP2SETUP_Z
#define CLIP2SETUP_Z_SIZE 73

#define CLIP2SETUP_Z_Z0_ROW                     0

#define CLIP2SETUP_Z_DZDX_ROW                   0

#define CLIP2SETUP_Z_DZDY_ROW                   0

#define CLIP2SETUP_Z_EXP_ROW                    0

#define CLIP2SETUP_Z_TID_ROW                    0


// Packet CLIP2ATRAST_TRAM
#define CLIP2ATRAST_TRAM_SIZE 270

#define CLIP2ATRAST_TRAM_C0_ROW                 0

#define CLIP2ATRAST_TRAM_C0_LO_ROW                      0

#define CLIP2ATRAST_TRAM_C0_HI_ROW                      0

#define CLIP2ATRAST_TRAM_C1_ROW                 0

#define CLIP2ATRAST_TRAM_C1_LO_ROW                      0

#define CLIP2ATRAST_TRAM_C1_HI_ROW                      0

#define CLIP2ATRAST_TRAM_C2_ROW                 0

#define CLIP2ATRAST_TRAM_C2_LO_ROW                      0

#define CLIP2ATRAST_TRAM_C2_HI_ROW                      0

#define CLIP2ATRAST_TRAM_C3_ROW                 0

#define CLIP2ATRAST_TRAM_C3_LO_ROW                      0

#define CLIP2ATRAST_TRAM_C3_HI_ROW                      0

#define CLIP2ATRAST_TRAM_TRAM_ROW_ROW                   0

#define CLIP2ATRAST_TRAM_WE_ROW                 0


// Packet TRAM_BF_HP
#define TRAM_BF_HP_SIZE 63

#define TRAM_BF_HP_DP0_MANT_ROW                 0

#define TRAM_BF_HP_DP1_MANT_ROW                 0

#define TRAM_BF_HP_P2_MANT_ROW                  0

#define TRAM_BF_HP_EXP_ROW                      0


// Packet TRAM_BF_LP
#define TRAM_BF_LP_SIZE 32

#define TRAM_BF_LP_DP0_MANT_ROW                 0

#define TRAM_BF_LP_DP1_MANT_ROW                 0

#define TRAM_BF_LP_P2_MANT_ROW                  0

#define TRAM_BF_LP_EXP_ROW                      0


// Packet TRAM_BF_ZP64
#define TRAM_BF_ZP64_SIZE 64

#define TRAM_BF_ZP64_DZDX_MANT_ROW                      0

#define TRAM_BF_ZP64_DZDY_MANT_ROW                      0

#define TRAM_BF_ZP64_Z0_ROW                     0

#define TRAM_BF_ZP64_EXP_ROW                    0


// Packet SETUP2QRAST_QR
#define SETUP2QRAST_QR_SIZE 40

#define SETUP2QRAST_QR_Q_ROW                    0

#define SETUP2QRAST_QR_Q_MANT_ROW                       0

#define SETUP2QRAST_QR_Q_SIGN_ROW                       0

#define SETUP2QRAST_QR_R_ROW                    0

#define SETUP2QRAST_QR_R_MANT_ROW                       0

#define SETUP2QRAST_QR_R_SIGN_ROW                       0


// Packet SETUP2QRAST
#define SETUP2QRAST_SIZE 110

#define SETUP2QRAST_DATA_ROW                    0

#define SETUP2QRAST_TAG_ROW                     0
#define SETUP2QRAST_TAG_REG                     (0)
#define SETUP2QRAST_TAG_Z                       (1)
#define SETUP2QRAST_TAG_C                       (2)
#define SETUP2QRAST_TAG_DCDX                    (3)
#define SETUP2QRAST_TAG_C_DCDX0                 (4)
#define SETUP2QRAST_TAG_C_DCDX1                 (5)
#define SETUP2QRAST_TAG_DCDY_DCDX0                      (6)
#define SETUP2QRAST_TAG_DCDY_DCDX1                      (7)
#define SETUP2QRAST_TAG_XY                      (8)


// Packet SETUP2QRAST_REG
#define SETUP2QRAST_REG_SIZE 110

#define SETUP2QRAST_REG_DATA_ROW                        0

#define SETUP2QRAST_REG_ADDR_ROW                        0

#define SETUP2QRAST_REG_BLK_NUM_ROW                     0
#define SETUP2QRAST_REG_BLK_NUM_CTL                     (0)
#define SETUP2QRAST_REG_BLK_NUM_IDX                     (1)
#define SETUP2QRAST_REG_BLK_NUM_VPE                     (2)
#define SETUP2QRAST_REG_BLK_NUM_SU                      (3)
#define SETUP2QRAST_REG_BLK_NUM_QR                      (4)
#define SETUP2QRAST_REG_BLK_NUM_PSEQ                    (5)
#define SETUP2QRAST_REG_BLK_NUM_AT                      (6)
#define SETUP2QRAST_REG_BLK_NUM_TEX                     (7)
#define SETUP2QRAST_REG_BLK_NUM_ALU                     (8)
#define SETUP2QRAST_REG_BLK_NUM_DW                      (9)
#define SETUP2QRAST_REG_BLK_NUM_FDC                     (10)
#define SETUP2QRAST_REG_BLK_NUM_GLB1                    (14)
#define SETUP2QRAST_REG_BLK_NUM_GLB2                    (15)

#define SETUP2QRAST_REG_BLK_ADDR_ROW                    0

#define SETUP2QRAST_REG_READ_ROW                        0

#define SETUP2QRAST_REG_CHANNEL_ROW                     0

#define SETUP2QRAST_REG_PD_ROW                  0

#define SETUP2QRAST_REG_TAG_ROW                 0
#define SETUP2QRAST_REG_TAG_REG                 (0)
#define SETUP2QRAST_REG_TAG_Z                   (1)
#define SETUP2QRAST_REG_TAG_C                   (2)
#define SETUP2QRAST_REG_TAG_DCDX                        (3)
#define SETUP2QRAST_REG_TAG_C_DCDX0                     (4)
#define SETUP2QRAST_REG_TAG_C_DCDX1                     (5)
#define SETUP2QRAST_REG_TAG_DCDY_DCDX0                  (6)
#define SETUP2QRAST_REG_TAG_DCDY_DCDX1                  (7)
#define SETUP2QRAST_REG_TAG_XY                  (8)


// Packet SETUP2QRAST_Z
#define SETUP2QRAST_Z_SIZE 110

#define SETUP2QRAST_Z_Z0_ROW                    0

#define SETUP2QRAST_Z_DZDX_ROW                  0

#define SETUP2QRAST_Z_DZDY_ROW                  0

#define SETUP2QRAST_Z_EXP_ROW                   0

#define SETUP2QRAST_Z_TID_ROW                   0

#define SETUP2QRAST_Z_PD_ROW                    0

#define SETUP2QRAST_Z_TAG_ROW                   0
#define SETUP2QRAST_Z_TAG_REG                   (0)
#define SETUP2QRAST_Z_TAG_Z                     (1)
#define SETUP2QRAST_Z_TAG_C                     (2)
#define SETUP2QRAST_Z_TAG_DCDX                  (3)
#define SETUP2QRAST_Z_TAG_C_DCDX0                       (4)
#define SETUP2QRAST_Z_TAG_C_DCDX1                       (5)
#define SETUP2QRAST_Z_TAG_DCDY_DCDX0                    (6)
#define SETUP2QRAST_Z_TAG_DCDY_DCDX1                    (7)
#define SETUP2QRAST_Z_TAG_XY                    (8)


// Packet SETUP2QRAST_C
#define SETUP2QRAST_C_SIZE 110

#define SETUP2QRAST_C_A_ROW                     0

#define SETUP2QRAST_C_B_ROW                     0

#define SETUP2QRAST_C_UPPER_IN_EDGE_ROW                 0

#define SETUP2QRAST_C_UPPER_OUT_EDGE_ROW                        0

#define SETUP2QRAST_C_LOWER_IN_EDGE_ROW                 0

#define SETUP2QRAST_C_LOWER_OUT_EDGE_ROW                        0

#define SETUP2QRAST_C_FACE_ROW                  0
#define SETUP2QRAST_C_FACE_FRONT                        (0)
#define SETUP2QRAST_C_FACE_BACK                 (1)

#define SETUP2QRAST_C_PD_ROW                    0

#define SETUP2QRAST_C_TAG_ROW                   0
#define SETUP2QRAST_C_TAG_REG                   (0)
#define SETUP2QRAST_C_TAG_Z                     (1)
#define SETUP2QRAST_C_TAG_C                     (2)
#define SETUP2QRAST_C_TAG_DCDX                  (3)
#define SETUP2QRAST_C_TAG_C_DCDX0                       (4)
#define SETUP2QRAST_C_TAG_C_DCDX1                       (5)
#define SETUP2QRAST_C_TAG_DCDY_DCDX0                    (6)
#define SETUP2QRAST_C_TAG_DCDY_DCDX1                    (7)
#define SETUP2QRAST_C_TAG_XY                    (8)


// Packet SETUP2QRAST_DCDX
#define SETUP2QRAST_DCDX_SIZE 110

#define SETUP2QRAST_DCDX_DADX_ROW                       0

#define SETUP2QRAST_DCDX_DBDX_ROW                       0

#define SETUP2QRAST_DCDX_DGDX_ROW                       0

#define SETUP2QRAST_DCDX_DGPDX_ROW                      0

#define SETUP2QRAST_DCDX_PD_ROW                 0

#define SETUP2QRAST_DCDX_TAG_ROW                        0
#define SETUP2QRAST_DCDX_TAG_REG                        (0)
#define SETUP2QRAST_DCDX_TAG_Z                  (1)
#define SETUP2QRAST_DCDX_TAG_C                  (2)
#define SETUP2QRAST_DCDX_TAG_DCDX                       (3)
#define SETUP2QRAST_DCDX_TAG_C_DCDX0                    (4)
#define SETUP2QRAST_DCDX_TAG_C_DCDX1                    (5)
#define SETUP2QRAST_DCDX_TAG_DCDY_DCDX0                 (6)
#define SETUP2QRAST_DCDX_TAG_DCDY_DCDX1                 (7)
#define SETUP2QRAST_DCDX_TAG_XY                 (8)


// Packet SETUP2QRAST_C_DCDX0
#define SETUP2QRAST_C_DCDX0_SIZE 110

#define SETUP2QRAST_C_DCDX0_A_DADX_ROW                  0

#define SETUP2QRAST_C_DCDX0_A_DADX_Q_ROW                        0

#define SETUP2QRAST_C_DCDX0_A_DADX_R_ROW                        0

#define SETUP2QRAST_C_DCDX0_B_DBDX_ROW                  0

#define SETUP2QRAST_C_DCDX0_B_DBDX_Q_ROW                        0

#define SETUP2QRAST_C_DCDX0_B_DBDX_R_ROW                        0

#define SETUP2QRAST_C_DCDX0_PD_ROW                      0

#define SETUP2QRAST_C_DCDX0_TAG_ROW                     0
#define SETUP2QRAST_C_DCDX0_TAG_REG                     (0)
#define SETUP2QRAST_C_DCDX0_TAG_Z                       (1)
#define SETUP2QRAST_C_DCDX0_TAG_C                       (2)
#define SETUP2QRAST_C_DCDX0_TAG_DCDX                    (3)
#define SETUP2QRAST_C_DCDX0_TAG_C_DCDX0                 (4)
#define SETUP2QRAST_C_DCDX0_TAG_C_DCDX1                 (5)
#define SETUP2QRAST_C_DCDX0_TAG_DCDY_DCDX0                      (6)
#define SETUP2QRAST_C_DCDX0_TAG_DCDY_DCDX1                      (7)
#define SETUP2QRAST_C_DCDX0_TAG_XY                      (8)


// Packet SETUP2QRAST_C_DCDX1
#define SETUP2QRAST_C_DCDX1_SIZE 110

#define SETUP2QRAST_C_DCDX1_G_DGDX_ROW                  0

#define SETUP2QRAST_C_DCDX1_G_DGDX_Q_ROW                        0

#define SETUP2QRAST_C_DCDX1_G_DGDX_R_ROW                        0

#define SETUP2QRAST_C_DCDX1_G_P_DGPDX_ROW                       0

#define SETUP2QRAST_C_DCDX1_G_P_DGPDX_Q_ROW                     0

#define SETUP2QRAST_C_DCDX1_G_P_DGPDX_R_ROW                     0

#define SETUP2QRAST_C_DCDX1_PD_ROW                      0

#define SETUP2QRAST_C_DCDX1_TAG_ROW                     0
#define SETUP2QRAST_C_DCDX1_TAG_REG                     (0)
#define SETUP2QRAST_C_DCDX1_TAG_Z                       (1)
#define SETUP2QRAST_C_DCDX1_TAG_C                       (2)
#define SETUP2QRAST_C_DCDX1_TAG_DCDX                    (3)
#define SETUP2QRAST_C_DCDX1_TAG_C_DCDX0                 (4)
#define SETUP2QRAST_C_DCDX1_TAG_C_DCDX1                 (5)
#define SETUP2QRAST_C_DCDX1_TAG_DCDY_DCDX0                      (6)
#define SETUP2QRAST_C_DCDX1_TAG_DCDY_DCDX1                      (7)
#define SETUP2QRAST_C_DCDX1_TAG_XY                      (8)


// Packet SETUP2QRAST_DCDY_DCDX0
#define SETUP2QRAST_DCDY_DCDX0_SIZE 110

#define SETUP2QRAST_DCDY_DCDX0_DADY_DADX_ROW                    0

#define SETUP2QRAST_DCDY_DCDX0_DADY_DADX_Q_ROW                  0

#define SETUP2QRAST_DCDY_DCDX0_DADY_DADX_R_ROW                  0

#define SETUP2QRAST_DCDY_DCDX0_DBDY_DBDX_ROW                    0

#define SETUP2QRAST_DCDY_DCDX0_DBDY_DBDX_Q_ROW                  0

#define SETUP2QRAST_DCDY_DCDX0_DBDY_DBDX_R_ROW                  0

#define SETUP2QRAST_DCDY_DCDX0_PD_ROW                   0

#define SETUP2QRAST_DCDY_DCDX0_TAG_ROW                  0
#define SETUP2QRAST_DCDY_DCDX0_TAG_REG                  (0)
#define SETUP2QRAST_DCDY_DCDX0_TAG_Z                    (1)
#define SETUP2QRAST_DCDY_DCDX0_TAG_C                    (2)
#define SETUP2QRAST_DCDY_DCDX0_TAG_DCDX                 (3)
#define SETUP2QRAST_DCDY_DCDX0_TAG_C_DCDX0                      (4)
#define SETUP2QRAST_DCDY_DCDX0_TAG_C_DCDX1                      (5)
#define SETUP2QRAST_DCDY_DCDX0_TAG_DCDY_DCDX0                   (6)
#define SETUP2QRAST_DCDY_DCDX0_TAG_DCDY_DCDX1                   (7)
#define SETUP2QRAST_DCDY_DCDX0_TAG_XY                   (8)


// Packet SETUP2QRAST_DCDY_DCDX1
#define SETUP2QRAST_DCDY_DCDX1_SIZE 110

#define SETUP2QRAST_DCDY_DCDX1_DGDY_DGDX_ROW                    0

#define SETUP2QRAST_DCDY_DCDX1_DGDY_DGDX_Q_ROW                  0

#define SETUP2QRAST_DCDY_DCDX1_DGDY_DGDX_R_ROW                  0

#define SETUP2QRAST_DCDY_DCDX1_DGPDY_DGPDX_ROW                  0

#define SETUP2QRAST_DCDY_DCDX1_DGPDY_DGPDX_Q_ROW                        0

#define SETUP2QRAST_DCDY_DCDX1_DGPDY_DGPDX_R_ROW                        0

#define SETUP2QRAST_DCDY_DCDX1_PD_ROW                   0

#define SETUP2QRAST_DCDY_DCDX1_TAG_ROW                  0
#define SETUP2QRAST_DCDY_DCDX1_TAG_REG                  (0)
#define SETUP2QRAST_DCDY_DCDX1_TAG_Z                    (1)
#define SETUP2QRAST_DCDY_DCDX1_TAG_C                    (2)
#define SETUP2QRAST_DCDY_DCDX1_TAG_DCDX                 (3)
#define SETUP2QRAST_DCDY_DCDX1_TAG_C_DCDX0                      (4)
#define SETUP2QRAST_DCDY_DCDX1_TAG_C_DCDX1                      (5)
#define SETUP2QRAST_DCDY_DCDX1_TAG_DCDY_DCDX0                   (6)
#define SETUP2QRAST_DCDY_DCDX1_TAG_DCDY_DCDX1                   (7)
#define SETUP2QRAST_DCDY_DCDX1_TAG_XY                   (8)


// Packet SETUP2QRAST_XY
#define SETUP2QRAST_XY_SIZE 110

#define SETUP2QRAST_XY_XMIN_ROW                 0

#define SETUP2QRAST_XY_XMAX_ROW                 0

#define SETUP2QRAST_XY_YMIN_ROW                 0

#define SETUP2QRAST_XY_YMAX_ROW                 0

#define SETUP2QRAST_XY_YMID_IN_ROW                      0

#define SETUP2QRAST_XY_YMID_OUT_ROW                     0

#define SETUP2QRAST_XY_PD_ROW                   0

#define SETUP2QRAST_XY_TAG_ROW                  0
#define SETUP2QRAST_XY_TAG_REG                  (0)
#define SETUP2QRAST_XY_TAG_Z                    (1)
#define SETUP2QRAST_XY_TAG_C                    (2)
#define SETUP2QRAST_XY_TAG_DCDX                 (3)
#define SETUP2QRAST_XY_TAG_C_DCDX0                      (4)
#define SETUP2QRAST_XY_TAG_C_DCDX1                      (5)
#define SETUP2QRAST_XY_TAG_DCDY_DCDX0                   (6)
#define SETUP2QRAST_XY_TAG_DCDY_DCDX1                   (7)
#define SETUP2QRAST_XY_TAG_XY                   (8)


// Packet SETUP2ATRAST_SLOPES
#define SETUP2ATRAST_SLOPES_SIZE 81

#define SETUP2ATRAST_SLOPES_DATA_ROW                    0

#define SETUP2ATRAST_SLOPES_C0_ROW                      0

#define SETUP2ATRAST_SLOPES_DCDX_ROW                    0

#define SETUP2ATRAST_SLOPES_DCDY_ROW                    0

#define SETUP2ATRAST_SLOPES_EXP_ROW                     0

#define SETUP2ATRAST_SLOPES_TID_ROW                     0

#define SETUP2ATRAST_SLOPES_TAG_ROW                     0
#define SETUP2ATRAST_SLOPES_TAG_ALPHA                   (0)
#define SETUP2ATRAST_SLOPES_TAG_BETA                    (1)
#define SETUP2ATRAST_SLOPES_TAG_GAMMA                   (2)


// Packet SETUP2ATRAST_SLOPES_C0
#define SETUP2ATRAST_SLOPES_C0_SIZE 31

#define SETUP2ATRAST_SLOPES_C0_C0_ROW                   0

#define SETUP2ATRAST_SLOPES_C0_SIGN_ROW                 0

#define SETUP2ATRAST_SLOPES_C0_INT_ROW                  0

#define SETUP2ATRAST_SLOPES_C0_FRAC_ROW                 0


// Packet SETUP2ATRAST_SLOPES_DCDXDY
#define SETUP2ATRAST_SLOPES_DCDXDY_SIZE 18

#define SETUP2ATRAST_SLOPES_DCDXDY_DCDX_ROW                     0

#define SETUP2ATRAST_SLOPES_DCDXDY_DCDY_ROW                     0

#define SETUP2ATRAST_SLOPES_DCDXDY_SIGN_ROW                     0

#define SETUP2ATRAST_SLOPES_DCDXDY_INT_ROW                      0

#define SETUP2ATRAST_SLOPES_DCDXDY_FRAC_ROW                     0


// Packet QRAST2CLIP
#define QRAST2CLIP_SIZE 6

#define QRAST2CLIP_TID_ROW                      0


// Packet PSEQ2QRAST
#define PSEQ2QRAST_SIZE 6

#define PSEQ2QRAST_TRI_ID_ROW                   0


// Packet GK2SETUP_RDAT
#define GK2SETUP_RDAT_SIZE 38

#define GK2SETUP_RDAT_DATA_ROW                  0

#define GK2SETUP_RDAT_CHANNEL_ROW                       0

#define GK2SETUP_RDAT_TYPE_ROW                  0
#define GK2SETUP_RDAT_TYPE_REGISTER                     (0)
#define GK2SETUP_RDAT_TYPE_RAISE                        (1)
#define GK2SETUP_RDAT_TYPE_REFCNT                       (2)
#define GK2SETUP_RDAT_TYPE_CTXSW_ACK                    (3)


// Packet BARY_C_PER_HI
#define BARY_C_PER_HI_SIZE 16

#define BARY_C_PER_HI_FRAC_ROW                  0

#define BARY_C_PER_HI_INT_ROW                   0

#define BARY_C_PER_HI_SIGN_ROW                  0


// Packet BARY_C_PER_LO
#define BARY_C_PER_LO_SIZE 12

#define BARY_C_PER_LO_FRAC_ROW                  0

#define BARY_C_PER_LO_INT_ROW                   0

#define BARY_C_PER_LO_SIGN_ROW                  0

#define RASTER_SB_WIDTH 7

// Packet RS_SB
#define RS_SB_SIZE 7

#define RS_SB_TAG_ROW                   0

#define RS_SB_KILL_PIX_ROW                      0

#define RS_SB_ODD_ROW                   0

#define RS_SB_SEQ_ROW                   0

#define RASTER_PACKET_WIDTH     80
#define RASTER_PACKET_ARG_WIDTH 20

// Packet RAST_R
#define RAST_R_SIZE 40

#define RAST_R_R_ROW                    0

#define RAST_R_BLUE_ROW                 0

#define RAST_R_GREEN_ROW                        0

#define RAST_R_RED_ROW                  0

#define RAST_R_ALPHA_ROW                        0

#define RAST_R_L10_ROW                  0

#define RAST_R_H10_ROW                  0

#define RAST_R_Z_ROW                    0

#define RAST_R_ALPHA_LOW_ROW                    0

#define RAST_R_GREEN_LOW_ROW                    0

#define RAST_R_TEX_LOD_ROW                      0

#define RAST_R_TEX_ROW                  0

#define RAST_R_TEX_LODF_ROW                     0

#define RAST_R_TEX_QUADLIN_ROW                  0

#define RAST_R_TEX_FORMAT_ROW                   0

#define RAST_R_TEX_T_FRAC_ROW                   0

#define RAST_R_TEX_S_FRAC_ROW                   0

#define RAST_R_TEX_SAME_T_ROW                   0

#define RAST_R_TEX_SAME_S_ROW                   0

#define RAST_R_TEX_BILIN_ROW                    0

#define RAST_R_TEX_ADR_BNK0_BIT3_ROW                    0

#define RAST_R_TEX_ADR_BNK1_BIT3_ROW                    0

#define RAST_R_TEX_ADR_BNK2_BIT3_ROW                    0

#define RAST_R_TEX_ADR_BNK3_BIT3_ROW                    0

#define RAST_R_TEX_IN_LINE_MSK_ROW                      0

#define RAST_R_TEX_UNUSED_ROW                   0


// Packet RAST_H
#define RAST_H_SIZE 10

#define RAST_H_FX10_ROW                 0

#define RAST_H_FRACTION_ROW                     0

#define RAST_H_INT_ROW                  0

#define RAST_H_SIGN_ROW                 0


// Packet PIXPKT
#define PIXPKT_SIZE 87

#define PIXPKT_SEQ_ROW                  0

#define PIXPKT_KILL_ROW                 0

#define PIXPKT_TAG_ROW                  0
#define PIXPKT_TAG_PIX                  (0)
#define PIXPKT_TAG_REG                  (1)

#define PIXPKT_X_ROW                    0

#define PIXPKT_RASTER_ROW                       0

#define PIXPKT_R0_ROW                   0

#define PIXPKT_R1_ROW                   0

#define PIXPKT_R2_ROW                   0

#define PIXPKT_R3_ROW                   0


// Packet GK_PIXPKT
#define GK_PIXPKT_SIZE 88

#define GK_PIXPKT_SBSTALLED_ROW                 0

#define GK_PIXPKT_SEQ_ROW                       0

#define GK_PIXPKT_KILL_ROW                      0

#define GK_PIXPKT_TAG_ROW                       0
#define GK_PIXPKT_TAG_PIX                       (0)
#define GK_PIXPKT_TAG_REG                       (1)

#define GK_PIXPKT_X_ROW                 0

#define GK_PIXPKT_RASTER_ROW                    0

#define GK_PIXPKT_R0_ROW                        0

#define GK_PIXPKT_R1_ROW                        0

#define GK_PIXPKT_R2_ROW                        0

#define GK_PIXPKT_R3_ROW                        0


// Packet REG_WRITE
#define REG_WRITE_SIZE 54

#define REG_WRITE_DATA_ROW                      0

#define REG_WRITE_ADDR_ROW                      0

#define REG_WRITE_BLK_ADDR_ROW                  0

#define REG_WRITE_BLK_NUM_ROW                   0
#define REG_WRITE_BLK_NUM_VTX                   (0)
#define REG_WRITE_BLK_NUM_SETUP                 (1)
#define REG_WRITE_BLK_NUM_RS                    (2)
#define REG_WRITE_BLK_NUM_GK                    (3)
#define REG_WRITE_BLK_NUM_DF                    (4)
#define REG_WRITE_BLK_NUM_ALU                   (5)
#define REG_WRITE_BLK_NUM_ALU0                  (6)
#define REG_WRITE_BLK_NUM_ALU1                  (7)
#define REG_WRITE_BLK_NUM_ALU2                  (8)
#define REG_WRITE_BLK_NUM_ALU3                  (9)
#define REG_WRITE_BLK_NUM_DW                    (10)
#define REG_WRITE_BLK_NUM_FDC                   (11)
#define REG_WRITE_BLK_NUM_GLB1                  (14)
#define REG_WRITE_BLK_NUM_GLB2                  (15)

#define REG_WRITE_GLB_ROW                       0

#define REG_WRITE_READ_ROW                      0

#define REG_WRITE_CHANNEL_ROW                   0

#define REG_WRITE_READ_DONE_ROW                 0

#define REG_WRITE_READ_CHAN_ROW                 0


// Packet SCOREBOARD_FLUSH
#define SCOREBOARD_FLUSH_SIZE 14

#define SCOREBOARD_FLUSH_MASK_ROW                       0

#define SCOREBOARD_FLUSH_HASH_ROW                       0

#define SCOREBOARD_FLUSH_VALID_ROW                      0

#define LWE397_GK_MAX_LOOP_CNT    8100
#define LWE397_GK_BLK_SB_WIDTH    64
#define LWE397_GK_BLK_SB_WD_INDEX 6
#define LWE397_GK_BLK_SB_DEPTH    32
#define LWE397_GK_BLK_SB_DP_INDEX 5
#define LW_TEX_PKTFIFO_DEPTH    200
#define LW_TEX_STUFFFIFO_DEPTH  200
#define LW_TEX_RESULTFIFO_DEPTH 200
#define LW_TEX_LODSKIPFIFO_DEPTH        32

// Packet TEXSTUFFFIFO
#define TEXSTUFFFIFO_SIZE 118

#define TEXSTUFFFIFO_Bnk2_mem_addr_s0_orig_ROW                  0

#define TEXSTUFFFIFO_Bnk0_mem_addr_s0_orig_ROW                  0

#define TEXSTUFFFIFO_align_Bnk02_ROW                    0

#define TEXSTUFFFIFO_align_Bnk13_ROW                    0

#define TEXSTUFFFIFO_s0_ROW                     0

#define TEXSTUFFFIFO_s1_ROW                     0

#define TEXSTUFFFIFO_t0_ROW                     0

#define TEXSTUFFFIFO_t1_ROW                     0

#define TEXSTUFFFIFO_awtw_last_ROW                      0

#define TEXSTUFFFIFO_awtw_ROW                   0

#define TEXSTUFFFIFO_awtw_weight_ROW                    0

#define TEXSTUFFFIFO_layout_ROW                 0

#define TEXSTUFFFIFO_surf_format_ROW                    0

#define TEXSTUFFFIFO_t_frac_ROW                 0

#define TEXSTUFFFIFO_s_frac_ROW                 0

#define TEXSTUFFFIFO_bilinear_ROW                       0

#define TEXSTUFFFIFO_in_line_add_msk_ROW                        0

#define TEXSTUFFFIFO_ul_bnk_ROW                 0

#define TEXSTUFFFIFO_Bnk3_miss_req_ROW                  0

#define TEXSTUFFFIFO_Bnk2_miss_req_ROW                  0

#define TEXSTUFFFIFO_Bnk1_miss_req_ROW                  0

#define TEXSTUFFFIFO_Bnk0_miss_req_ROW                  0

#define TEXSTUFFFIFO_Bnk0Bnk2_sm2big_mem_addr_ROW                       0

#define TEXSTUFFFIFO_Bnk1Bnk3_sm2big_mem_addr_ROW                       0

#define TEXSTUFFFIFO_Bnk2_mem_addr_s0_ROW                       0

#define TEXSTUFFFIFO_Bnk0_mem_addr_s0_ROW                       0

#define TEXSTUFFFIFO_Bnk0_entry_ROW                     0

#define TEXSTUFFFIFO_Bnk0_hit_ROW                       0

#define TEXSTUFFFIFO_Bnk1_entry_ROW                     0

#define TEXSTUFFFIFO_Bnk1_hit_ROW                       0

#define TEXSTUFFFIFO_Bnk2_entry_ROW                     0

#define TEXSTUFFFIFO_Bnk2_hit_ROW                       0

#define TEXSTUFFFIFO_Bnk3_entry_ROW                     0

#define TEXSTUFFFIFO_Bnk3_hit_ROW                       0

#define TEXSTUFFFIFO_merge_bnk02_ROW                    0

#define TEXSTUFFFIFO_merge_bnk13_ROW                    0

#define TEXSTUFFFIFO_instr_ROW                  0

#define TEXSTUFFFIFO_in_line_address_ROW                        0


// Packet TEXRESULTFIFO
#define TEXRESULTFIFO_SIZE 43

#define TEXRESULTFIFO_lwrrent_float_ROW                 0

#define TEXRESULTFIFO_float_format_ROW                  0

#define TEXRESULTFIFO_sloc_ROW                  0

#define TEXRESULTFIFO_tloc_ROW                  0

#define TEXRESULTFIFO_data_ROW                  0


// Packet TEX_RAWTEX
#define TEX_RAWTEX_SIZE 60

#define TEX_RAWTEX_RAW4_ROW                     0

#define TEX_RAWTEX_RAW8_ROW                     0

#define TEX_RAWTEX_RAW16_ROW                    0

#define TEX_RAWTEX_RAW32_ROW                    0

#define TEX_RAWTEX_COLOR0_ROW                   0

#define TEX_RAWTEX_COLOR1_ROW                   0

#define TEX_RAWTEX_RGB_IDX_ROW                  0

#define TEX_RAWTEX_DXT1_ROW                     0

#define TEX_RAWTEX_A_DXT3_ROW                   0

#define TEX_RAWTEX_DXT3_ROW                     0

#define TEX_RAWTEX_A_IDX_ROW                    0

#define TEX_RAWTEX_A1LESS_ROW                   0

#define TEX_RAWTEX_ALOWER_ROW                   0

#define TEX_RAWTEX_ADELTA_ROW                   0

#define TEX_RAWTEX_DXT5_ROW                     0

#define TEX_RAWTEX_ETC_S1_R1_INDIV_ROW                  0

#define TEX_RAWTEX_ETC_S1_G1_INDIV_ROW                  0

#define TEX_RAWTEX_ETC_S1_B1_INDIV_ROW                  0

#define TEX_RAWTEX_ETC_S2_R2_INDIV_ROW                  0

#define TEX_RAWTEX_ETC_S2_G2_INDIV_ROW                  0

#define TEX_RAWTEX_ETC_S2_B2_INDIV_ROW                  0

#define TEX_RAWTEX_ETC_S1_R1_DIFF_ROW                   0

#define TEX_RAWTEX_ETC_S1_G1_DIFF_ROW                   0

#define TEX_RAWTEX_ETC_S1_B1_DIFF_ROW                   0

#define TEX_RAWTEX_ETC_S2_DR2_DIFF_ROW                  0

#define TEX_RAWTEX_ETC_S2_DG2_DIFF_ROW                  0

#define TEX_RAWTEX_ETC_S2_DB2_DIFF_ROW                  0

#define TEX_RAWTEX_ETC_FLIPBIT_ROW                      0

#define TEX_RAWTEX_ETC_DIFF_ROW                 0

#define TEX_RAWTEX_ETC_CW1_ROW                  0

#define TEX_RAWTEX_ETC_CW2_ROW                  0

#define TEX_RAWTEX_ETC_CW_M0_ROW                        0

#define TEX_RAWTEX_ETC_CW_M1_ROW                        0

#define TEX_RAWTEX_ETC_CW_M2_ROW                        0

#define TEX_RAWTEX_ETC_CW_M3_ROW                        0

#define TEX_RAWTEX_ETC_ROW                      0

#define TEX_RAWTEX_ETC_COLOR_ROW                        0

#define TEX_RAWTEX_ETC_PIXIDX_MSB_ROW                   0

#define TEX_RAWTEX_ETC_PIXIDX_LSB_ROW                   0

#define TEX_RAWTEX_ETC_SUBBLOCK_ROW                     0

#define TEX_RAWTEX_ETC3_A_ROW                   0

#define TEX_RAWTEX_ETC3_ROW                     0

#define TEX_RAWTEX_ETC5_A_IDX_ROW                       0

#define TEX_RAWTEX_ETC5_A1LESS_ROW                      0

#define TEX_RAWTEX_ETC5_ALOWER_ROW                      0

#define TEX_RAWTEX_ETC5_ADELTA_ROW                      0

#define TEX_RAWTEX_ETC5_ROW                     0

#define TEX_RAWTEX_A_IDX_1_ROW                  0

#define TEX_RAWTEX_A1LESS_1_ROW                 0

#define TEX_RAWTEX_ALOWER_1_ROW                 0

#define TEX_RAWTEX_ADELTA_1_ROW                 0

#define LWE397_TEX_INLINE_ADDR_SIZE       7
#define LWE397_TEX_INLINE_S_ADDR_SIZE     4
#define LWE397_TEX_INLINE_T_ADDR_SIZE     3
#define DW_CACHE_LINE_SIZE      128

// Packet DW_PAYLD
#define DW_PAYLD_SIZE 80

#define DW_PAYLD_BLUE_0_ROW                     0

#define DW_PAYLD_GREEN_0_ROW                    0

#define DW_PAYLD_RED_0_ROW                      0

#define DW_PAYLD_ALPHA_0_ROW                    0

#define DW_PAYLD_RED_1_ROW                      0

#define DW_PAYLD_GREEN_1_ROW                    0

#define DW_PAYLD_BLUE_1_ROW                     0

#define DW_PAYLD_ALPHA_1_ROW                    0


// Packet DW_COLOR_FMT10
#define DW_COLOR_FMT10_SIZE 10

#define DW_COLOR_FMT10_SIGN_ROW                 0

#define DW_COLOR_FMT10_INT_ROW                  0

#define DW_COLOR_FMT10_FRACTION_ROW                     0

#define DW_COLOR_FMT10_COLOR_ROW                        0


// Packet DW_MEM_PKT
#define DW_MEM_PKT_SIZE 179

#define DW_MEM_PKT_MEM_PAYLD_0_ROW                      0

#define DW_MEM_PKT_MEM_PAYLD_1_ROW                      0

#define DW_MEM_PKT_MEM_PAYLD_2_ROW                      0

#define DW_MEM_PKT_MEM_PAYLD_3_ROW                      0

#define DW_MEM_PKT_ADDR_ROW                     0

#define DW_MEM_PKT_BE_ROW                       0

#define DW_MEM_PKT_PKT_TYPE_ROW                 0
#define DW_MEM_PKT_PKT_TYPE_CVAL                        (0)
#define DW_MEM_PKT_PKT_TYPE_ZVAL                        (1)


// Packet DW_SB_PKT
#define DW_SB_PKT_SIZE 17

#define DW_SB_PKT_TYPE_ROW                      0
#define DW_SB_PKT_TYPE_CVAL                     (0)
#define DW_SB_PKT_TYPE_ZVAL                     (1)
#define DW_SB_PKT_TYPE_BOTH                     (2)
#define DW_SB_PKT_TYPE_PS                       (3)
#define DW_SB_PKT_TYPE_RAISE                    (4)

#define DW_SB_PKT_MASK_ROW                      0

#define DW_SB_PKT_HASH_ROW                      0

#define DW_SB_PKT_VALID_ROW                     0


// Packet DW_SB_PS_CACHE
#define DW_SB_PS_CACHE_SIZE 8

#define DW_SB_PS_CACHE_PIX0_ROW                 0

#define DW_SB_PS_CACHE_PIX1_ROW                 0

#define DW_SB_PS_CACHE_PIX2_ROW                 0

#define DW_SB_PS_CACHE_PIX3_ROW                 0

#define DW_SB_PS_CACHE_PIX4_ROW                 0

#define DW_SB_PS_CACHE_PIX5_ROW                 0

#define DW_SB_PS_CACHE_PIX6_ROW                 0

#define DW_SB_PS_CACHE_PIX7_ROW                 0


// Packet DW_SBFIFO
#define DW_SBFIFO_SIZE 18

#define DW_SBFIFO_WRITE_C_ROW                   0

#define DW_SBFIFO_WRITE_Z_ROW                   0

#define DW_SBFIFO_MASK_ROW                      0

#define DW_SBFIFO_HASH_Y_ROW                    0

#define DW_SBFIFO_HASH_X_ROW                    0


// Packet DW_SB_MASK
#define DW_SB_MASK_SIZE 4

#define DW_SB_MASK_PR0_ROW                      0

#define DW_SB_MASK_PR1_ROW                      0

#define DW_SB_MASK_PR2_ROW                      0

#define DW_SB_MASK_PR3_ROW                      0


// Packet FAKE_VID_PIXEL
#define FAKE_VID_PIXEL_SIZE 54

#define FAKE_VID_PIXEL_CMD_ROW                  0
#define FAKE_VID_PIXEL_CMD_PIX                  (0)
#define FAKE_VID_PIXEL_CMD_LASTH_PIX                    (1)
#define FAKE_VID_PIXEL_CMD_LASTV_PIX                    (2)

#define FAKE_VID_PIXEL_X_ROW                    0

#define FAKE_VID_PIXEL_Y_ROW                    0

#define FAKE_VID_PIXEL_RED_ROW                  0

#define FAKE_VID_PIXEL_GREEN_ROW                        0

#define FAKE_VID_PIXEL_BLUE_ROW                 0

#define IBUF_BITS       3
#define OBUF_BITS       4
#define IBUF_RAM_BITS   6
#define OBUF_RAM_BITS   6

// Packet IDX2CLIP
#define IDX2CLIP_SIZE 49

#define IDX2CLIP_DATA_ROW                       0

#define IDX2CLIP_ADDR_ROW                       0

#define IDX2CLIP_READ_ROW                       0

#define IDX2CLIP_CHANNEL_ROW                    0


// Packet PSEQ2IDX
#define PSEQ2IDX_SIZE 38

#define PSEQ2IDX_DATA_ROW                       0

#define PSEQ2IDX_CHANNEL_ROW                    0

#define PSEQ2IDX_TYPE_ROW                       0
#define PSEQ2IDX_TYPE_READ                      (0)
#define PSEQ2IDX_TYPE_RAISE                     (1)
#define PSEQ2IDX_TYPE_REFCNT                    (2)


// Packet IDX2VPE
#define IDX2VPE_SIZE 190

#define IDX2VPE_DATA_0_ROW                      0

#define IDX2VPE_DATA_1_ROW                      0

#define IDX2VPE_DATA_2_ROW                      0

#define IDX2VPE_DATA_3_ROW                      0

#define IDX2VPE_ADDR_ROW                        0

#define IDX2VPE_VALID_ROW                       0

#define IDX2VPE_MASK_ROW                        0

#define IDX2VPE_VABCTXADDR_ROW                  0

#define IDX2VPE_COMMAND_ROW                     0
#define IDX2VPE_COMMAND_NOP                     (0)
#define IDX2VPE_COMMAND_VAB                     (1)
#define IDX2VPE_COMMAND_XFPR                    (2)
#define IDX2VPE_COMMAND_PASSTHR                 (5)
#define IDX2VPE_COMMAND_CONST                   (11)

#define IDX2VPE_LAUNCH_ROW                      0

#define IDX2VPE_PRI_DEBUG_ROW                   0

#define IDX2VPE_PRI_CTRL_ROW                    0

#define IDX2VPE_OBUF_AVAIL_ROW                  0

#define IDX2VPE_STEN_ROW                        0

#define VPE_GE_PROGRAM_CTRL_BUNDLE      204
#define VPE_ATTRIBINPUTMASK_BUNDLE      205
#define VPE_ATTRIBOUTPUTMASK_BUNDLE     206
#define VPE_GEOM_STALL_BUNDLE   224
#define VPE_GE_PROGRAM_TIMEOUT_BUNDLE   227
#define VPE_GE_CONST_LIMITS_BUNDLE      231
#define VPE_BRANCHBITS_BUNDLE   237
#define VPE_PIPE_NOP_BUNDLE     497

// Packet VPE_GE_PROGRAM_CTRL
#define VPE_GE_PROGRAM_CTRL_SIZE 32

#define VPE_GE_PROGRAM_CTRL_PROGRAM_START_ROW                   0

#define VPE_GE_PROGRAM_CTRL_SIGNED_ZERO_COMPARE_ROW                     0
#define VPE_GE_PROGRAM_CTRL_SIGNED_ZERO_COMPARE_UNEQUAL                 (0)
#define VPE_GE_PROGRAM_CTRL_SIGNED_ZERO_COMPARE_EQUAL                   (1)

#define VPE_GE_PROGRAM_CTRL_MODE_ROW                    0
#define VPE_GE_PROGRAM_CTRL_MODE_FIXED                  (0)
#define VPE_GE_PROGRAM_CTRL_MODE_PROGRAM                        (2)
#define VPE_GE_PROGRAM_CTRL_MODE_PROGRAM_V2                     (3)
#define VPE_GE_PROGRAM_CTRL_MODE_PROGRAM_V3                     (1)


// Packet VPE_GEOM_STALL
#define VPE_GEOM_STALL_SIZE 17

#define VPE_GEOM_STALL_STALL_ROW                        0

#define VPE_GEOM_STALL_FLUSH_ROW                        0


// Packet VPE_GE_PROGRAM_TIMEOUT
#define VPE_GE_PROGRAM_TIMEOUT_SIZE 17

#define VPE_GE_PROGRAM_TIMEOUT_COUNT_ROW                        0

#define VPE_GE_PROGRAM_TIMEOUT_REGISTER_COUNT_ROW                       0
#define VPE_GE_PROGRAM_TIMEOUT_REGISTER_COUNT_RC32                      (0)
#define VPE_GE_PROGRAM_TIMEOUT_REGISTER_COUNT_RC48                      (1)


// Packet VPE_GE_CONST_LIMITS
#define VPE_GE_CONST_LIMITS_SIZE 26

#define VPE_GE_CONST_LIMITS_MIN_ROW                     0

#define VPE_GE_CONST_LIMITS_MAX_ROW                     0


// Packet VPE2IDX
#define VPE2IDX_SIZE 7

#define VPE2IDX_IDLE_ROW                        0

#define VPE2IDX_RETIRE_MODE_ROW                 0

#define VPE2IDX_RETIRE_VALID_ROW                        0

#define VPE2IDX_RETIRE_IBUF_ROW                 0


// Packet IDX2VPE_READADDR
#define IDX2VPE_READADDR_SIZE 9

#define IDX2VPE_READADDR_ADDR_ROW                       0

#define IDX2VPE_READADDR_WHICH_ROW                      0
#define IDX2VPE_READADDR_WHICH_CONST                    (0)
#define IDX2VPE_READADDR_WHICH_INSN                     (1)


// Packet VPE2IDX_READDATA
#define VPE2IDX_READDATA_SIZE 128

#define VPE2IDX_READDATA_DATA_ROW                       0


// Packet CLIP2IDX
#define CLIP2IDX_SIZE 32

#define CLIP2IDX_LWLL_ROW                       0

#define CLIP2IDX_READ_ROW                       0


// Packet VPE2CLIP
#define VPE2CLIP_SIZE 9

#define VPE2CLIP_DONE_ROW                       0

#define VPE2CLIP_OBUF_STRIDE_ROW                        0


// Packet CLIP2VPE_ATTRADDR
#define CLIP2VPE_ATTRADDR_SIZE 11

#define CLIP2VPE_ATTRADDR_READADDR_ROW                  0

#define CLIP2VPE_ATTRADDR_OBUFADDR_ROW                  0

#define CLIP2VPE_ATTRADDR_POS_VALID_ROW                 0


// Packet VPE2CLIP_ATTRDATA
#define VPE2CLIP_ATTRDATA_SIZE 149

#define VPE2CLIP_ATTRDATA_ATTR_0_ROW                    0

#define VPE2CLIP_ATTRDATA_ATTR_1_ROW                    0

#define VPE2CLIP_ATTRDATA_ATTR_2_ROW                    0

#define VPE2CLIP_ATTRDATA_ATTR_3_ROW                    0

#define VPE2CLIP_ATTRDATA_WRITTEN_ROW                   0

#define VPE2CLIP_ATTRDATA_ATTRIBUTES_ROW                        0

#define VPE2CLIP_ATTRDATA_POS_VALID_ROW                 0


// Packet PIXSHADERPKT
#define PIXSHADERPKT_SIZE 100

#define PIXSHADERPKT_STATE_ROW                  0
#define PIXSHADERPKT_STATE_REGISTER                     (0)
#define PIXSHADERPKT_STATE_SPANSTART                    (1)
#define PIXSHADERPKT_STATE_Z_PASS                       (2)
#define PIXSHADERPKT_STATE_Z_FAIL                       (3)
#define PIXSHADERPKT_STATE_S_FAIL                       (4)
#define PIXSHADERPKT_STATE_KILL                 (5)
#define PIXSHADERPKT_STATE_NON_CENTER                   (6)

#define PIXSHADERPKT_DBG_ROW                    0

#define PIXSHADERPKT_PAYLOAD_ROW                        0


// Packet PIXSHADERPKT_PIX
#define PIXSHADERPKT_PIX_SIZE 100

#define PIXSHADERPKT_PIX_STATE_ROW                      0
#define PIXSHADERPKT_PIX_STATE_Z_PASS                   (2)
#define PIXSHADERPKT_PIX_STATE_Z_FAIL                   (3)
#define PIXSHADERPKT_PIX_STATE_S_FAIL                   (4)
#define PIXSHADERPKT_PIX_STATE_KILL                     (5)
#define PIXSHADERPKT_PIX_STATE_NON_CENTER                       (6)

#define PIXSHADERPKT_PIX_DBG_ROW                        0

#define PIXSHADERPKT_PIX_INTERLEAVE_ROW                 0

#define PIXSHADERPKT_PIX_SEQ_ROW                        0

#define PIXSHADERPKT_PIX_SB_ROW                 0

#define PIXSHADERPKT_PIX_CX_ROW                 0
#define PIXSHADERPKT_PIX_CX_EXELWTE                     (0)
#define PIXSHADERPKT_PIX_CX_NO_EXELWTE                  (1)

#define PIXSHADERPKT_PIX_VCAA_ROW                       0

#define PIXSHADERPKT_PIX_PREDICATE_ROW                  0

#define PIXSHADERPKT_PIX_DELTA_ROW                      0

#define PIXSHADERPKT_PIX_RASTER_ROW                     0

#define PIXSHADERPKT_PIX_R0_ROW                 0

#define PIXSHADERPKT_PIX_R0_HI_ROW                      0

#define PIXSHADERPKT_PIX_R0_LO_ROW                      0

#define PIXSHADERPKT_PIX_R1_ROW                 0

#define PIXSHADERPKT_PIX_R1_HI_ROW                      0

#define PIXSHADERPKT_PIX_R1_LO_ROW                      0

#define PIXSHADERPKT_PIX_R2_ROW                 0

#define PIXSHADERPKT_PIX_R2_HI_ROW                      0

#define PIXSHADERPKT_PIX_R2_LO_ROW                      0

#define PIXSHADERPKT_PIX_R3_ROW                 0

#define PIXSHADERPKT_PIX_R3_HI_ROW                      0

#define PIXSHADERPKT_PIX_R3_LO_ROW                      0


// Packet PIXSHADERPKT_SPAN
#define PIXSHADERPKT_SPAN_SIZE 100

#define PIXSHADERPKT_SPAN_STATE_ROW                     0
#define PIXSHADERPKT_SPAN_STATE_SPANSTART                       (1)

#define PIXSHADERPKT_SPAN_DBG_ROW                       0

#define PIXSHADERPKT_SPAN_SEQ_ROW                       0

#define PIXSHADERPKT_SPAN_QID_ROW                       0

#define PIXSHADERPKT_SPAN_HEAD_ROW                      0

#define PIXSHADERPKT_SPAN_SPILL_ROW                     0

#define PIXSHADERPKT_SPAN_FACE_ROW                      0
#define PIXSHADERPKT_SPAN_FACE_FRONT                    (0)
#define PIXSHADERPKT_SPAN_FACE_BACK                     (1)

#define PIXSHADERPKT_SPAN_TRI_ID_ROW                    0

#define PIXSHADERPKT_SPAN_SPAN_X_ROW                    0

#define PIXSHADERPKT_SPAN_SPAN_Y_ROW                    0


// Packet PIXSHADERPKT_REG
#define PIXSHADERPKT_REG_SIZE 100

#define PIXSHADERPKT_REG_STATE_ROW                      0
#define PIXSHADERPKT_REG_STATE_REGISTER                 (0)

#define PIXSHADERPKT_REG_DBG_ROW                        0

#define PIXSHADERPKT_REG_DATA1_ROW                      0

#define PIXSHADERPKT_REG_CHANNEL_ROW                    0

#define PIXSHADERPKT_REG_DATA1_WR_EN_ROW                        0

#define PIXSHADERPKT_REG_READ_ROW                       0

#define PIXSHADERPKT_REG_DW_ST_ROW                      0

#define PIXSHADERPKT_REG_BLK_NUM_ROW                    0
#define PIXSHADERPKT_REG_BLK_NUM_CTL                    (0)
#define PIXSHADERPKT_REG_BLK_NUM_IDX                    (1)
#define PIXSHADERPKT_REG_BLK_NUM_VPE                    (2)
#define PIXSHADERPKT_REG_BLK_NUM_SU                     (3)
#define PIXSHADERPKT_REG_BLK_NUM_QR                     (4)
#define PIXSHADERPKT_REG_BLK_NUM_PSEQ                   (5)
#define PIXSHADERPKT_REG_BLK_NUM_AT                     (6)
#define PIXSHADERPKT_REG_BLK_NUM_TEX                    (7)
#define PIXSHADERPKT_REG_BLK_NUM_ALU                    (8)
#define PIXSHADERPKT_REG_BLK_NUM_DW                     (9)
#define PIXSHADERPKT_REG_BLK_NUM_FDC                    (10)
#define PIXSHADERPKT_REG_BLK_NUM_GLB1                   (14)
#define PIXSHADERPKT_REG_BLK_NUM_GLB2                   (15)

#define PIXSHADERPKT_REG_BLK_ADDR_ROW                   0

#define PIXSHADERPKT_REG_ADDR_ROW                       0

#define PIXSHADERPKT_REG_DATA_ROW                       0

#define BARY_DC_WIDTH   18
#define BARY_C_WIDTH    37
#define BARY_W_DC_WIDTH 22
#define BARY_W_C_WIDTH  38
#define QRAST_SL_RAM_WIDTH      228

// Packet LWBEMAP_FACE
#define LWBEMAP_FACE_SIZE 3

#define LWBEMAP_FACE_FACE_ROW                   0
#define LWBEMAP_FACE_FACE_POSITIVE_X                    (0)
#define LWBEMAP_FACE_FACE_NEGATIVE_X                    (1)
#define LWBEMAP_FACE_FACE_POSITIVE_Y                    (2)
#define LWBEMAP_FACE_FACE_NEGATIVE_Y                    (3)
#define LWBEMAP_FACE_FACE_POSITIVE_Z                    (4)
#define LWBEMAP_FACE_FACE_NEGATIVE_Z                    (5)

#define LW_FDC_PSEQ_CLIENT_LATENCY      128
#define LW_FDC_CLIENT_LATENCY   200
#define LW_FDC_MC_AW    32
#define LW_FDC_MC_WDLOG2        5
#define LW_FDC_MC_MW    256
#define LW_FDC_ADDR_WIDTH       29
#define LW_FDC_DATA_WIDTH       64
#define LW_FDC_BE_WIDTH 8
#define LW_FDC_OFFS_WIDTH       2
#define LW_FDC_TAG_WIDTH        27
#define LW_FDC_USE_SPLIT_MC_ODD_EVEN    0
#define LW_FDC_XMEM_SPLITBIT    26
#define LW_FDC_XMEM_SELECT      0
#define LW_FDC_ODD_EVEN_SPLITBIT        0
#define LW_FDC_ODD_EVEN_SELECT  0
#define LW_FDC_L2_SIZE  32768
#define LW_FDC_L2_LINE_SIZE     256
#define LW_FDC_L2_LINES 128
#define LW_FDC_L2_SETS  1
#define LW_FDC_L2_WAYS  128
#define LW_FDC_L2_MEM_COLUMNS   4
#define LW_FDC_L2_MEM_SEGMENTS  2
#define LW_FDC_L2_MEM_SEGMENTS_LOG2     1
#define LW_FDC_L2_MEM_BANKS     8
#define LW_FDC_READ_CLEAN_CNT_WIDTH     1
#define LW_FDC_REPL_LFSR_SIZE   16
#define LW_FDC_REPL_POL 106513

// Packet LW_FDC_REQ_ADDRESS
#define LW_FDC_REQ_ADDRESS_SIZE 29

#define LW_FDC_REQ_ADDRESS_TAG_ROW                      0

#define LW_FDC_REQ_ADDRESS_OFFSET_ROW                   0

#define LW_FDC_L1RREQ_MAXREQ    4
#define LW_FDC_L1RREQ_MAXREQ_CNT_WIDTH  2
#define LW_FDC_L1RREQ_SV_QFIFO_DEPTH    100
#define LW_FDC_L1RREQ_Z_QFIFO_DEPTH     100
#define LW_FDC_L1RREQ_P_QFIFO_DEPTH     128

// Packet LW_FDC_L1RREQ_PENDQUEUE
#define LW_FDC_L1RREQ_PENDQUEUE_SIZE 13

#define LW_FDC_L1RREQ_PENDQUEUE_RELEASE_ROW                     0

#define LW_FDC_L1RREQ_PENDQUEUE_REQ_MODE_ROW                    0
#define LW_FDC_L1RREQ_PENDQUEUE_REQ_MODE_PIXEL                  (0)
#define LW_FDC_L1RREQ_PENDQUEUE_REQ_MODE_REGISTER                       (1)

#define LW_FDC_L1RREQ_PENDQUEUE_LINE_MODE_ROW                   0
#define LW_FDC_L1RREQ_PENDQUEUE_LINE_MODE_L64BIT                        (0)
#define LW_FDC_L1RREQ_PENDQUEUE_LINE_MODE_L128BIT                       (1)

#define LW_FDC_L1RREQ_PENDQUEUE_REG_ENC_ADDRESS_ROW                     0

#define LW_FDC_L1RREQ_PENDQUEUE_LINE_ROW                        0

#define LW_FDC_L1RREQ_PENDQUEUE_OFFSET_ROW                      0

#define LW_FDC_L1RREQ_PENDQUEUE_COUNT_ROW                       0

#define LW_FDC_L1WREQ_PENDQUEUE_BE_BASE 75
#define LW_FDC_L1WREQ_PENDQUEUE_DATA_BASE       11

// Packet LW_FDC_L1WREQ_PENDQUEUE
#define LW_FDC_L1WREQ_PENDQUEUE_SIZE 83

#define LW_FDC_L1WREQ_PENDQUEUE_BE_ROW                  0

#define LW_FDC_L1WREQ_PENDQUEUE_REG_DATA_ROW                    0

#define LW_FDC_L1WREQ_PENDQUEUE_DATA_ROW                        0

#define LW_FDC_L1WREQ_PENDQUEUE_RELEASE_ROW                     0

#define LW_FDC_L1WREQ_PENDQUEUE_REQ_MODE_ROW                    0
#define LW_FDC_L1WREQ_PENDQUEUE_REQ_MODE_PIXEL                  (0)
#define LW_FDC_L1WREQ_PENDQUEUE_REQ_MODE_REGISTER                       (1)

#define LW_FDC_L1WREQ_PENDQUEUE_LINE_MODE_ROW                   0
#define LW_FDC_L1WREQ_PENDQUEUE_LINE_MODE_L64BIT                        (0)
#define LW_FDC_L1WREQ_PENDQUEUE_LINE_MODE_L128BIT                       (1)

#define LW_FDC_L1WREQ_PENDQUEUE_REG_ENC_ADDRESS_ROW                     0

#define LW_FDC_L1WREQ_PENDQUEUE_LINE_ROW                        0

#define LW_FDC_L1WREQ_PENDQUEUE_OFFSET_ROW                      0

#define LW_FDC_L1WREQH_PENDQUEUE_BE_BASE        75
#define LW_FDC_L1WREQH_PENDQUEUE_DATA_BASE      11

// Packet LW_FDC_L1WREQH_PENDQUEUE
#define LW_FDC_L1WREQH_PENDQUEUE_SIZE 112

#define LW_FDC_L1WREQH_PENDQUEUE_ADDRESS_ROW                    0

#define LW_FDC_L1WREQH_PENDQUEUE_BE_ROW                 0

#define LW_FDC_L1WREQH_PENDQUEUE_REG_DATA_ROW                   0

#define LW_FDC_L1WREQH_PENDQUEUE_DATA_ROW                       0

#define LW_FDC_L1WREQH_PENDQUEUE_RELEASE_ROW                    0

#define LW_FDC_L1WREQH_PENDQUEUE_REQ_MODE_ROW                   0
#define LW_FDC_L1WREQH_PENDQUEUE_REQ_MODE_PIXEL                 (0)
#define LW_FDC_L1WREQH_PENDQUEUE_REQ_MODE_REGISTER                      (1)

#define LW_FDC_L1WREQH_PENDQUEUE_LINE_MODE_ROW                  0
#define LW_FDC_L1WREQH_PENDQUEUE_LINE_MODE_L64BIT                       (0)
#define LW_FDC_L1WREQH_PENDQUEUE_LINE_MODE_L128BIT                      (1)

#define LW_FDC_L1WREQH_PENDQUEUE_REG_ENC_ADDRESS_ROW                    0

#define LW_FDC_L1WREQH_PENDQUEUE_LINE_ROW                       0

#define LW_FDC_L1WREQH_PENDQUEUE_OFFSET_ROW                     0


// Packet LW_FDC_REQ_REG_ADDR
#define LW_FDC_REQ_REG_ADDR_SIZE 12

#define LW_FDC_REQ_REG_ADDR_BLK_NUM_ROW                 0

#define LW_FDC_REQ_REG_ADDR_BLK_ADDR_ROW                        0


// Packet LW_FDC_REQ_REG_ADDR_ENC
#define LW_FDC_REQ_REG_ADDR_ENC_SIZE 9

#define LW_FDC_REQ_REG_ADDR_ENC_BLK_NUM_ROW                     0
#define LW_FDC_REQ_REG_ADDR_ENC_BLK_NUM_FDC                     (0)
#define LW_FDC_REQ_REG_ADDR_ENC_BLK_NUM_GLOBAL                  (1)

#define LW_FDC_REQ_REG_ADDR_ENC_BLK_ADDR_ROW                    0
#define LW_FDC_REQ_REG_ADDR_ENC_BLK_ADDR_GLOBAL_LW_MCCIF                        (0)
#define LW_FDC_REQ_REG_ADDR_ENC_BLK_ADDR_OTHER                  (15)

#define LW_FDC_MCQ_RD_QFIFO_WIDTH       8
#define LW_FDC_MCQ_WR_QFIFO_WIDTH       7
#define LW_FDC_MCQ_WR_DRAM_PEND_QFIFO_DEPTH     32
#define LW_FDC_MCQ_RD_DRAM_PEND_QFIFO_DEPTH     32

// Packet LW_FDC_MC_QUEUE
#define LW_FDC_MC_QUEUE_SIZE 7

#define LW_FDC_MC_QUEUE_LINE_ROW                        0

#define LW_FDC_SB_QFIFO_DEPTH   16

// Packet LW_FDC_L2LINESTAT
#define LW_FDC_L2LINESTAT_SIZE 9

#define LW_FDC_L2LINESTAT_REQCNT_ROW                    0

#define LW_FDC_L2LINESTAT_DIRTY_ROW                     0

#define LW_FDC_L2LINESTAT_LOCKED_ROW                    0

#define LW_FDC_L2LINESTAT_MCDONE_ROW                    0

#define LW_FDC_L2LINESTAT_COMPLETE_ROW                  0

#define LW_FDC_L2LINESTAT_PERSISTENT_ROW                        0

#define LW_FDC_L2LINESTAT_READCLEAN_ROW                 0


// Packet LW_FDC_RD_CLIENT_REQ
#define LW_FDC_RD_CLIENT_REQ_SIZE 28

#define LW_FDC_RD_CLIENT_REQ_PERSISTENT_ROW                     0

#define LW_FDC_RD_CLIENT_REQ_TAG_ROW                    0


// Packet LW_FDC_WR_CLIENT_REQ
#define LW_FDC_WR_CLIENT_REQ_SIZE 29

#define LW_FDC_WR_CLIENT_REQ_READCLEAN_ROW                      0

#define LW_FDC_WR_CLIENT_REQ_PERSISTENT_ROW                     0

#define LW_FDC_WR_CLIENT_REQ_TAG_ROW                    0


// Packet LW_FDC_MCQ_READ
#define LW_FDC_MCQ_READ_SIZE 35

#define LW_FDC_MCQ_READ_BUF_SWAP_ROW                    0

#define LW_FDC_MCQ_READ_LINE_ROW                        0

#define LW_FDC_MCQ_READ_ADDRESS_ROW                     0


// Packet LW_FDC_MCQ_RDATA
#define LW_FDC_MCQ_RDATA_SIZE 264

#define LW_FDC_MCQ_RDATA_BUF_SWAP_ROW                   0

#define LW_FDC_MCQ_RDATA_DATA_ROW                       0

#define LW_FDC_MCQ_RDATA_LINE_ROW                       0


// Packet LW_FDC_MCQ_WRITE
#define LW_FDC_MCQ_WRITE_SIZE 322

#define LW_FDC_MCQ_WRITE_DATA_ROW                       0

#define LW_FDC_MCQ_WRITE_BE_ROW                 0

#define LW_FDC_MCQ_WRITE_LINE_ROW                       0

#define LW_FDC_MCQ_WRITE_ADDRESS_ROW                    0


// Packet LW_FDC_SB_CHECK
#define LW_FDC_SB_CHECK_SIZE 32

#define LW_FDC_SB_CHECK_BYTES_PER_PIXEL_ROW                     0
#define LW_FDC_SB_CHECK_BYTES_PER_PIXEL_BPP1                    (0)
#define LW_FDC_SB_CHECK_BYTES_PER_PIXEL_BPP2                    (1)
#define LW_FDC_SB_CHECK_BYTES_PER_PIXEL_BPP4                    (2)
#define LW_FDC_SB_CHECK_BYTES_PER_PIXEL_BPP8                    (3)

#define LW_FDC_SB_CHECK_QUAD_Y_ROW                      0

#define LW_FDC_SB_CHECK_QUAD_X_ROW                      0

#define LW_FDC_SB_CHECK_BYTEMASK_ROW                    0


// Packet LW_FDC_SB_PKT
#define LW_FDC_SB_PKT_SIZE 30

#define LW_FDC_SB_PKT_PAYLOAD_ROW                       0

#define LW_FDC_SB_PKT_QUAD_Y_ROW                        0

#define LW_FDC_SB_PKT_QUAD_X_ROW                        0

#define LW_FDC_SB_PKT_MASK_ROW                  0


// Packet LW_FDC_RREQ
#define LW_FDC_RREQ_SIZE 32

#define LW_FDC_RREQ_PERSISTENT_ROW                      0

#define LW_FDC_RREQ_REG_UNUSED_ROW                      0

#define LW_FDC_RREQ_REG_BLK_NUM_ROW                     0
#define LW_FDC_RREQ_REG_BLK_NUM_CTL                     (0)
#define LW_FDC_RREQ_REG_BLK_NUM_IDX                     (1)
#define LW_FDC_RREQ_REG_BLK_NUM_VPE                     (2)
#define LW_FDC_RREQ_REG_BLK_NUM_SU                      (3)
#define LW_FDC_RREQ_REG_BLK_NUM_QR                      (4)
#define LW_FDC_RREQ_REG_BLK_NUM_PSEQ                    (5)
#define LW_FDC_RREQ_REG_BLK_NUM_AT                      (6)
#define LW_FDC_RREQ_REG_BLK_NUM_TEX                     (7)
#define LW_FDC_RREQ_REG_BLK_NUM_ALU                     (8)
#define LW_FDC_RREQ_REG_BLK_NUM_DW                      (9)
#define LW_FDC_RREQ_REG_BLK_NUM_FDC                     (10)
#define LW_FDC_RREQ_REG_BLK_NUM_GLB1                    (14)
#define LW_FDC_RREQ_REG_BLK_NUM_GLB2                    (15)

#define LW_FDC_RREQ_REG_BLK_ADDR_ROW                    0

#define LW_FDC_RREQ_REG_ADDR_ROW                        0

#define LW_FDC_RREQ_TAG_ROW                     0

#define LW_FDC_RREQ_OFFSET_ROW                  0

#define LW_FDC_RREQ_ADDRESS_ROW                 0

#define LW_FDC_RREQ_LINE_MODE_ROW                       0
#define LW_FDC_RREQ_LINE_MODE_L64BIT                    (0)
#define LW_FDC_RREQ_LINE_MODE_L128BIT                   (1)

#define LW_FDC_RREQ_REQ_MODE_ROW                        0
#define LW_FDC_RREQ_REQ_MODE_PIXEL                      (0)
#define LW_FDC_RREQ_REQ_MODE_REGISTER                   (1)


// Packet LW_FDC_READ_DATA
#define LW_FDC_READ_DATA_SIZE 64

#define LW_FDC_READ_DATA_REG_UNUSED_ROW                 0

#define LW_FDC_READ_DATA_REG_DATA_ROW                   0

#define LW_FDC_READ_DATA_DATA_ROW                       0

#define LW_FDC_WREQ_DATA_BASE   31
#define LW_FDC_WREQ_BE_BASE     95

// Packet LW_FDC_WREQ
#define LW_FDC_WREQ_SIZE 133

#define LW_FDC_WREQ_SB_BYTES_PER_PIXEL_ROW                      0
#define LW_FDC_WREQ_SB_BYTES_PER_PIXEL_BPP1                     (0)
#define LW_FDC_WREQ_SB_BYTES_PER_PIXEL_BPP2                     (1)
#define LW_FDC_WREQ_SB_BYTES_PER_PIXEL_BPP4                     (2)
#define LW_FDC_WREQ_SB_BYTES_PER_PIXEL_BPP8                     (3)

#define LW_FDC_WREQ_SB_QUAD_Y_ROW                       0

#define LW_FDC_WREQ_SB_QUAD_X_ROW                       0

#define LW_FDC_WREQ_SB_MASK_ROW                 0

#define LW_FDC_WREQ_SB_CLEAR_ROW                        0

#define LW_FDC_WREQ_READ_CLEAN_ROW                      0

#define LW_FDC_WREQ_PERSISTENT_ROW                      0

#define LW_FDC_WREQ_BE_ROW                      0

#define LW_FDC_WREQ_REG_DATA_UNUSED_ROW                 0

#define LW_FDC_WREQ_REG_DATA_ROW                        0

#define LW_FDC_WREQ_DATA_ROW                    0

#define LW_FDC_WREQ_REG_UNUSED_ROW                      0

#define LW_FDC_WREQ_REG_BLK_NUM_ROW                     0
#define LW_FDC_WREQ_REG_BLK_NUM_CTL                     (0)
#define LW_FDC_WREQ_REG_BLK_NUM_IDX                     (1)
#define LW_FDC_WREQ_REG_BLK_NUM_VPE                     (2)
#define LW_FDC_WREQ_REG_BLK_NUM_SU                      (3)
#define LW_FDC_WREQ_REG_BLK_NUM_QR                      (4)
#define LW_FDC_WREQ_REG_BLK_NUM_PSEQ                    (5)
#define LW_FDC_WREQ_REG_BLK_NUM_AT                      (6)
#define LW_FDC_WREQ_REG_BLK_NUM_TEX                     (7)
#define LW_FDC_WREQ_REG_BLK_NUM_ALU                     (8)
#define LW_FDC_WREQ_REG_BLK_NUM_DW                      (9)
#define LW_FDC_WREQ_REG_BLK_NUM_FDC                     (10)
#define LW_FDC_WREQ_REG_BLK_NUM_GLB1                    (14)
#define LW_FDC_WREQ_REG_BLK_NUM_GLB2                    (15)

#define LW_FDC_WREQ_REG_BLK_ADDR_ROW                    0

#define LW_FDC_WREQ_REG_ADDR_ROW                        0

#define LW_FDC_WREQ_TAG_ROW                     0

#define LW_FDC_WREQ_OFFSET_ROW                  0

#define LW_FDC_WREQ_ADDRESS_ROW                 0

#define LW_FDC_WREQ_LINE_MODE_ROW                       0
#define LW_FDC_WREQ_LINE_MODE_L64BIT                    (0)
#define LW_FDC_WREQ_LINE_MODE_L128BIT                   (1)

#define LW_FDC_WREQ_REQ_MODE_ROW                        0
#define LW_FDC_WREQ_REQ_MODE_PIXEL                      (0)
#define LW_FDC_WREQ_REQ_MODE_REGISTER                   (1)

#define SETUP2QRAST_INPUTFIFO_DEPTH     1
#define QRAST2PSEQ_INPUTFIFO_DEPTH      1
#define DWR2PSEQ_INPUTFIFO_DEPTH        5
#define PSEQ2ATRAST_INPUTFIFO_DEPTH     1
#define ATRAST2TEX_INPUTFIFO_DEPTH      1
#define TEX2ALU_INPUTFIFO_DEPTH 1
#define ALU2DWR_INPUTFIFO_DEPTH 1
#define PSEQ_DATAFETCH_FIFO_DEPTH       112
#define PSEQ_SB_FIFO_DEPTH      4
#define PSEQ_IDX_FIFO_DEPTH     4

// Packet PSEQ_SIDEBAND
#define PSEQ_SIDEBAND_SIZE 4

#define PSEQ_SIDEBAND_DEST_ROW                  0
#define PSEQ_SIDEBAND_DEST_PASS_THROUGH                 (0)
#define PSEQ_SIDEBAND_DEST_UPPER_PIPE                   (1)
#define PSEQ_SIDEBAND_DEST_SPILL                        (2)
#define PSEQ_SIDEBAND_DEST_GATHER                       (3)

#define PSEQ_SIDEBAND_BUFFER1_ROW                       0

#define PSEQ_SIDEBAND_BUFFER0_ROW                       0


// Packet PSEQ_DATAFETCH
#define PSEQ_DATAFETCH_SIZE 104

#define PSEQ_DATAFETCH_PSEQ_SIDEBAND_ROW                        0

#define PSEQ_DATAFETCH_PIXSHADERPKT_ROW                 0


// Packet PIXSHADERPKT_GATHER
#define PIXSHADERPKT_GATHER_SIZE 100

#define PIXSHADERPKT_GATHER_DATA_ROW                    0

#define PIXSHADERPKT_GATHER_BYTE_OFFSET_ROW                     0


// Packet PIXSHADERPKT_MOP
#define PIXSHADERPKT_MOP_SIZE 100

#define PIXSHADERPKT_MOP_DATA_ROW                       0

#define PIXSHADERPKT_MOP_PC_ROW                 0

#define PIXSHADERPKT_MOP_SPILL_ROW                      0

#define PIXSHADERPKT_MOP_START_SEQ_ROW                  0


// Packet ATRAST_W_FLOAT
#define ATRAST_W_FLOAT_SIZE 25

#define ATRAST_W_FLOAT_MANT_ROW                 0

#define ATRAST_W_FLOAT_EXP_ROW                  0

#define ATRAST_W_FLOAT_SIGN_ROW                 0


// Packet ATRAST_AB_FLOAT
#define ATRAST_AB_FLOAT_SIZE 22

#define ATRAST_AB_FLOAT_MANT_ROW                        0

#define ATRAST_AB_FLOAT_EXP_ROW                 0

#define ATRAST_AB_FLOAT_SIGN_ROW                        0


// Packet ATRAST_AB_BLK_FLT
#define ATRAST_AB_BLK_FLT_SIZE 34

#define ATRAST_AB_BLK_FLT_AB_HP_MANT_ROW                        0

#define ATRAST_AB_BLK_FLT_AB_LP_MANT_ROW                        0

#define ATRAST_AB_BLK_FLT_EXP_ROW                       0

#define ATRAST_HP_IPA_REDUCE_BITS       12
#define LW_GR3D_VPE_CMDQ_RAM_DEPTH      8
#define LW_GR3D_VPE_CMDQ_WIDTH  18
#define LW_GR3D_VPE_DPATH_WIDTH 32
#define LW_GR3D_VPE_NUM_COMPONENTS      4
#define LW_GR3D_VPE_DPATH_BUS_WIDTH     128
#define LW_GR3D_VPE_CTX_SIZE    256
#define LW_GR3D_VPE_PROGRAM_LENGTH      256
#define LW_GR3D_VPE_CTX_ADDR_BITS       10
#define LW_GR3D_VPE_CMD_ADDR_BITS       10
#define LW_GR3D_VPE_CC_BITS     3
#define LW_GR3D_VPE_CC_REG_WIDTH        12
#define LW_GR3D_VPE_XFSIDBAND_WIDTH     17
#define LW_GR3D_VPE_MODE_BITS   1
#define LW_GR3D_VPE_INSTR_LIMIT 65536
#define LW_GR3D_VPE_INSTR_LIMIT_BITS    16
#define LW_GR3D_VPE_PRO_ADDR_BITS       8
#define LW_GR3D_VPE_PROGRAM_LD_WIDTH    127
#define LW_GR3D_VPE_THREAD_BITS 1
#define LW_GR3D_VPE_INSTR_OPR_TYPE_ILWAL        0
#define LW_GR3D_VPE_INSTR_OPR_TYPE_POS  0
#define LW_GR3D_VPE_INSTR_OPR_TYPE_REG  1
#define LW_GR3D_VPE_INSTR_OPR_TYPE_BUF  2
#define LW_GR3D_VPE_INSTR_OPR_TYPE_CTX  3
#define LW_GR3D_VPE_INSTR_EXTR_X        0
#define LW_GR3D_VPE_INSTR_EXTR_Y        1
#define LW_GR3D_VPE_INSTR_EXTR_Z        2
#define LW_GR3D_VPE_INSTR_EXTR_W        3
#define LW_GR3D_VPE_INSTR_CC_FALSE      0
#define LW_GR3D_VPE_INSTR_CC_LT 1
#define LW_GR3D_VPE_INSTR_CC_EQ 2
#define LW_GR3D_VPE_INSTR_CC_LE 3
#define LW_GR3D_VPE_INSTR_CC_GT 4
#define LW_GR3D_VPE_INSTR_CC_NE 5
#define LW_GR3D_VPE_INSTR_CC_GE 6
#define LW_GR3D_VPE_INSTR_CC_TRUE       7
#define LW_GR3D_VPE_INSTR_V_NOP 0
#define LW_GR3D_VPE_INSTR_V_MOV 1
#define LW_GR3D_VPE_INSTR_V_MUL 2
#define LW_GR3D_VPE_INSTR_V_ADD 3
#define LW_GR3D_VPE_INSTR_V_MAD 4
#define LW_GR3D_VPE_INSTR_V_DP3 5
#define LW_GR3D_VPE_INSTR_V_DPH 6
#define LW_GR3D_VPE_INSTR_V_DP4 7
#define LW_GR3D_VPE_INSTR_V_DST 8
#define LW_GR3D_VPE_INSTR_V_MIN 9
#define LW_GR3D_VPE_INSTR_V_MAX 10
#define LW_GR3D_VPE_INSTR_V_SLT 11
#define LW_GR3D_VPE_INSTR_V_SGE 12
#define LW_GR3D_VPE_INSTR_V_ARL 13
#define LW_GR3D_VPE_INSTR_V_FRC 14
#define LW_GR3D_VPE_INSTR_V_FLR 15
#define LW_GR3D_VPE_INSTR_V_SEQ 16
#define LW_GR3D_VPE_INSTR_V_SFL 17
#define LW_GR3D_VPE_INSTR_V_SGT 18
#define LW_GR3D_VPE_INSTR_V_SLE 19
#define LW_GR3D_VPE_INSTR_V_SNE 20
#define LW_GR3D_VPE_INSTR_V_STR 21
#define LW_GR3D_VPE_INSTR_V_SSG 22
#define LW_GR3D_VPE_INSTR_V_ARR 23
#define LW_GR3D_VPE_INSTR_V_MVA 24
#define LW_GR3D_VPE_INSTR_V_TXL 25
#define LW_GR3D_VPE_INSTR_V_PSH 26
#define LW_GR3D_VPE_INSTR_V_POP 27
#define LW_GR3D_VPE_INSTR_V_RSV0        28
#define LW_GR3D_VPE_INSTR_V_RSV1        29
#define LW_GR3D_VPE_INSTR_V_RSV2        30
#define LW_GR3D_VPE_INSTR_V_RSV3        31
#define LW_GR3D_VPE_INSTR_S_NOP 0
#define LW_GR3D_VPE_INSTR_S_MOV 1
#define LW_GR3D_VPE_INSTR_S_RCP 2
#define LW_GR3D_VPE_INSTR_S_RCC 3
#define LW_GR3D_VPE_INSTR_S_RSQ 4
#define LW_GR3D_VPE_INSTR_S_EXP 5
#define LW_GR3D_VPE_INSTR_S_LOG 6
#define LW_GR3D_VPE_INSTR_S_LIT 7
#define LW_GR3D_VPE_INSTR_S_BRA 8
#define LW_GR3D_VPE_INSTR_S_BRI 9
#define LW_GR3D_VPE_INSTR_S_CLA 10
#define LW_GR3D_VPE_INSTR_S_CLI 11
#define LW_GR3D_VPE_INSTR_S_RET 12
#define LW_GR3D_VPE_INSTR_S_LG2 13
#define LW_GR3D_VPE_INSTR_S_EX2 14
#define LW_GR3D_VPE_INSTR_S_SIN 15
#define LW_GR3D_VPE_INSTR_S_COS 16
#define LW_GR3D_VPE_INSTR_S_BRB 17
#define LW_GR3D_VPE_INSTR_S_CLB 18
#define LW_GR3D_VPE_INSTR_S_PSH 19
#define LW_GR3D_VPE_INSTR_S_POP 20
#define LW_GR3D_VPE_INSTR_S_RSV0        21
#define LW_GR3D_VPE_INSTR_S_RSV1        22
#define LW_GR3D_VPE_INSTR_S_RSV2        23
#define LW_GR3D_VPE_INSTR_S_RSV3        24
#define LW_GR3D_VPE_INSTR_S_RSV4        25
#define LW_GR3D_VPE_INSTR_S_RSV5        26
#define LW_GR3D_VPE_INSTR_S_RSV6        27
#define LW_GR3D_VPE_INSTR_S_RSV7        28
#define LW_GR3D_VPE_INSTR_S_RSV8        29
#define LW_GR3D_VPE_INSTR_S_RSV9        30
#define LW_GR3D_VPE_INSTR_S_RSVA        31
#define LW_GR3D_VPE_RANKINE_REG_NOP     31
#define LW_GR3D_VPE_INSTR_SEL_SCALAR    0
#define LW_GR3D_VPE_INSTR_SEL_VECTOR    1
#define LW_GR3D_VPE_PROG_TYPE_FFU       0
#define LW_GR3D_VPE_PROG_TYPE_VS1       2
#define LW_GR3D_VPE_PROG_TYPE_VS2       3
#define LW_GR3D_VPE_PROG_TYPE_VS3       1
#define LW_GR3D_VPE_CTL_ASEL_WIDTH      3
#define LW_GR3D_VPE_CTL_ASEL_IPOS       0
#define LW_GR3D_VPE_CTL_ASEL_IBUF       1
#define LW_GR3D_VPE_CTL_ASEL_CTX        2
#define LW_GR3D_VPE_CTL_ASEL_REG        4
#define LW_GR3D_VPE_CTL_ASEL_ADD_RSLT   5
#define LW_GR3D_VPE_CTL_ASEL_MUL_RSLT   6
#define LW_GR3D_VPE_CTL_ASEL_SCA_RSLT   7
#define LW_GR3D_VPE_CTL_BSEL_WIDTH      3
#define LW_GR3D_VPE_CTL_BSEL_IPOS       0
#define LW_GR3D_VPE_CTL_BSEL_IBUF       1
#define LW_GR3D_VPE_CTL_BSEL_CTX        2
#define LW_GR3D_VPE_CTL_BSEL_REG        4
#define LW_GR3D_VPE_CTL_BSEL_ADD_RSLT   5
#define LW_GR3D_VPE_CTL_BSEL_MUL_RSLT   6
#define LW_GR3D_VPE_CTL_BSEL_SCA_RSLT   7
#define LW_GR3D_VPE_CTL_CSEL_WIDTH      3
#define LW_GR3D_VPE_CTL_CSEL_IPOS       0
#define LW_GR3D_VPE_CTL_CSEL_IBUF       1
#define LW_GR3D_VPE_CTL_CSEL_CTX        2
#define LW_GR3D_VPE_CTL_CSEL_REG        4
#define LW_GR3D_VPE_CTL_CSEL_ADD_RSLT   5
#define LW_GR3D_VPE_CTL_CSEL_MUL_RSLT   6
#define LW_GR3D_VPE_CTL_CSEL_SCA_RSLT   7
#define LW_GR3D_VPE_CTL_ADD_SEL_ADD_RSLT        2
#define LW_GR3D_VPE_CTL_ADD_SEL_SCA_RSLT        3
#define LW_GR3D_VPE_CTL_ADD_SEL_PASS    0
#define LW_GR3D_VPE_CTL_IBUF_RADDR_POS  0
#define LW_GR3D_VPE_CTL_IBUF_RADDR_WT   1
#define LW_GR3D_VPE_CTL_IBUF_RADDR_NRL  2
#define LW_GR3D_VPE_CTL_IBUF_RADDR_C0   3
#define LW_GR3D_VPE_CTL_IBUF_RADDR_C1   4
#define LW_GR3D_VPE_CTL_IBUF_RADDR_FOG  5
#define LW_GR3D_VPE_CTL_IBUF_RADDR_PS   6
#define LW_GR3D_VPE_CTL_IBUF_RADDR_C2   7
#define LW_GR3D_VPE_CTL_IBUF_RADDR_C3   8
#define LW_GR3D_VPE_CTL_IBUF_RADDR_T0   8
#define LW_GR3D_VPE_CTL_IBUF_RADDR_T1   9
#define LW_GR3D_VPE_CTL_IBUF_RADDR_T2   10
#define LW_GR3D_VPE_CTL_IBUF_RADDR_T3   11
#define LW_GR3D_VPE_CTL_IBUF_RADDR_T4   12
#define LW_GR3D_VPE_CTL_IBUF_RADDR_T5   13
#define LW_GR3D_VPE_CTL_IBUF_RADDR_T6   14
#define LW_GR3D_VPE_CTL_IBUF_RADDR_T7   15
#define LW_GR3D_VPE_CTL_MUL_OP_MULT     0
#define LW_GR3D_VPE_CTL_MUL_OP_DIST     1
#define LW_GR3D_VPE_CTL_MUL_OP_PASA     2
#define LW_GR3D_VPE_CTL_MUL_OP_MULP     3
#define LW_GR3D_VPE_CTL_MUL_OP_MIN      4
#define LW_GR3D_VPE_CTL_MUL_OP_MAX      5
#define LW_GR3D_VPE_CTL_MUL_OP_SLT      6
#define LW_GR3D_VPE_CTL_MUL_OP_SGE      7
#define LW_GR3D_VPE_CTL_MUL_OP_FLR      8
#define LW_GR3D_VPE_CTL_MUL_OP_FRC      9
#define LW_GR3D_VPE_CTL_MUL_OP_SEQ      10
#define LW_GR3D_VPE_CTL_MUL_OP_SFL      11
#define LW_GR3D_VPE_CTL_MUL_OP_SNE      12
#define LW_GR3D_VPE_CTL_MUL_OP_SSG      13
#define LW_GR3D_VPE_CTL_MUL_OP_STR      14
#define LW_GR3D_VPE_CTL_MUL_OP_SGT      15
#define LW_GR3D_VPE_CTL_MUL_OP_SLE      16
#define LW_GR3D_VPE_CTL_ADD_OP_ADDT     0
#define LW_GR3D_VPE_CTL_ADD_OP_SUM3     2
#define LW_GR3D_VPE_CTL_ADD_OP_SUM4     3
#define LW_GR3D_VPE_CTL_ADD_OP_PASA     5
#define LW_GR3D_VPE_CTL_ADD_OP_PASB     6
#define LW_GR3D_VPE_CTL_SCA_OP_NOP      0
#define LW_GR3D_VPE_CTL_SCA_OP_ILW      1
#define LW_GR3D_VPE_CTL_SCA_OP_CILW     2
#define LW_GR3D_VPE_CTL_SCA_OP_ISQ      3
#define LW_GR3D_VPE_CTL_SCA_OP_MOV      4
#define LW_GR3D_VPE_CTL_SCA_OP_EXP      5
#define LW_GR3D_VPE_CTL_SCA_OP_LOG      6
#define LW_GR3D_VPE_CTL_SCA_OP_PWR      7
#define LW_GR3D_VPE_CTL_SCA_OP_SIN      8
#define LW_GR3D_VPE_CTL_SCA_OP_COS      9
#define LW_GR3D_VPE_CTL_SCA_OP_EX2      10
#define LW_GR3D_VPE_CTL_SCA_OP_LG2      11
#define LW_GR3D_VPE_CTL_REG_WMSK_W      0
#define LW_GR3D_VPE_CTL_REG_WMSK_Z      1
#define LW_GR3D_VPE_CTL_REG_WMSK_Y      2
#define LW_GR3D_VPE_CTL_REG_WMSK_X      3
#define LW_GR3D_VPE_CTL_RAM_WMSK_W      0
#define LW_GR3D_VPE_CTL_RAM_WMSK_Z      1
#define LW_GR3D_VPE_CTL_RAM_WMSK_Y      2
#define LW_GR3D_VPE_CTL_RAM_WMSK_X      3
#define LW_GR3D_VPE_UC012_TAG   5
#define LW_GR3D_VPE_UC345_TAG   6
#define LW_GR3D_VPE_OBUF_POS    0
#define LW_GR3D_VPE_OBUF_BDIFF  1
#define LW_GR3D_VPE_OBUF_BSPEC  2
#define LW_GR3D_VPE_OBUF_DIFF   3
#define LW_GR3D_VPE_OBUF_SPEC   4
#define LW_GR3D_VPE_OBUF_FPU    5
#define LW_GR3D_VPE_OBUF_UCP    6
#define LW_GR3D_VPE_OBUF_RSVD   7
#define LW_GR3D_VPE_OBUF_T0     8
#define LW_GR3D_VPE_OBUF_T1     9
#define LW_GR3D_VPE_OBUF_T2     10
#define LW_GR3D_VPE_OBUF_T3     11
#define LW_GR3D_VPE_OBUF_T4     12
#define LW_GR3D_VPE_OBUF_T5     13
#define LW_GR3D_VPE_OBUF_T6     14
#define LW_GR3D_VPE_OBUF_T7     15
#define LW_GR3D_VPE_OBUF_FOGW   16
#define LW_GR3D_VPE_OBUF_UC0    17
#define LW_GR3D_VPE_OBUF_UC1    18
#define LW_GR3D_VPE_OBUF_UC2    19
#define LW_GR3D_VPE_OBUF_UC3    20
#define LW_GR3D_VPE_OBUF_UC4    21
#define LW_GR3D_VPE_OBUF_UC5    22
#define LW_GR3D_VPE_OBUF_FOG    23
#define LW_GR3D_VPE_OBUF_PS     24
#define LW_GR3D_VPE_OBUF_NOP    31
#define LW_GR3D_VPE_CTL_OBUF_POS        0
#define LW_GR3D_VPE_OBUF_C2     1
#define LW_GR3D_VPE_OBUF_C3     2
#define LW_GR3D_VPE_OBUF_C0     3
#define LW_GR3D_VPE_OBUF_C1     4
#define XF_INSTR_OPCODE_WIDTH   10
#define XF_INSTR_CTX_WIDTH      8
#define XF_INSTR_IBUF_WIDTH     4
#define XF_INSTR_RA_WIDTH       4
#define XF_INSTR_RB_WIDTH       4
#define XF_INSTR_RC_WIDTH       4
#define XF_INSTR_RT_WIDTH       4
#define XF_INSTR_OUT_WIDTH      9
#define XF_INSTR_OPR_TYPE_CTX   3
#define XF_INSTR_OPR_TYPE_BUF   2
#define XF_INSTR_OPR_TYPE_REG   1
#define XF_INSTR_OPR_TYPE_ILWAL 0
#define XF_INSTR_OPR_TYPE_POS   0
#define XF_INSTR_EXTR_X 0
#define XF_INSTR_EXTR_Y 1
#define XF_INSTR_EXTR_Z 2
#define XF_INSTR_EXTR_W 3
#define XF_INSTR_OPCODE_WIDTH   10
#define XF_INSTR_CTX_WIDTH      8
#define XF_INSTR_IBUF_WIDTH     4
#define XF_INSTR_RA_WIDTH       4
#define XF_INSTR_RB_WIDTH       4
#define XF_INSTR_RC_WIDTH       4
#define XF_INSTR_EXTR_WIDTH     2
#define XF_INSTR_RT_WIDTH       4
#define XF_INSTR_OUT_WIDTH      9
#define XF_INSTR_CC_FALSE       0
#define XF_INSTR_CC_LT  1
#define XF_INSTR_CC_LT  1
#define XF_INSTR_CC_EQ  2
#define XF_INSTR_CC_LE  3
#define XF_INSTR_CC_GT  4
#define XF_INSTR_CC_NE  5
#define XF_INSTR_CC_GE  6
#define XF_INSTR_CC_TRUE        7
#define XF_V_NOP        0
#define XF_V_MOV        1
#define XF_V_MUL        2
#define XF_V_ADD        3
#define XF_V_MAD        4
#define XF_V_DP3        5
#define XF_V_DPH        6
#define XF_V_DP4        7
#define XF_V_DST        8
#define XF_V_MIN        9
#define XF_V_MAX        10
#define XF_V_SLT        11
#define XF_V_SGE        12
#define XF_V_ARL        13
#define XF_V_FRC        14
#define XF_V_FLR        15
#define XF_V_SEQ        16
#define XF_V_SFL        17
#define XF_V_SGT        18
#define XF_V_SLE        19
#define XF_V_SNE        20
#define XF_V_STR        21
#define XF_V_SSG        22
#define XF_V_ARR        23
#define XF_V_MVA        24
#define XF_V_RSV0       25
#define XF_V_RSV1       26
#define XF_V_RSV2       27
#define XF_V_RSV3       28
#define XF_V_RSV4       29
#define XF_V_RSV5       30
#define XF_V_RSV6       31
#define XF_S_NOP        0
#define XF_S_MOV        1
#define XF_S_RCP        2
#define XF_S_RCC        3
#define XF_S_RSQ        4
#define XF_S_EXP        5
#define XF_S_LOG        6
#define XF_S_LIT        7
#define XF_S_BRA        8
#define XF_S_BRI        9
#define XF_S_CLA        10
#define XF_S_CLI        11
#define XF_S_RET        12
#define XF_S_LG2        13
#define XF_S_EX2        14
#define XF_S_SIN        15
#define XF_S_COS        16
#define XF_S_RSV0       17
#define XF_S_RSV1       18
#define XF_S_RSV2       19
#define XF_S_RSV3       20
#define XF_S_RSV4       21
#define XF_S_RSV5       22
#define XF_S_RSV6       23
#define XF_S_RSV7       24
#define XF_S_RSV8       25
#define XF_S_RSV9       26
#define XF_S_RSVa       27
#define XF_S_RSVb       28
#define XF_S_RSVc       29
#define XF_S_RSVd       30
#define XF_S_RSVe       31
#define LW_GR3D_VPE_BUNDLE_ADDR_BITS    9
#define LW_GR3D_VPE_XFCMD_WIDTH 5
#define LW_GR3D_VPE_CICO_BUS_WIDTH      65
#define LW_GR3D_VPE_CTX_LWRIE_USER      512
#define LW_GR3D_VPE_CTX_LWRIE_PRIV      32
#define LW_GR3D_VPE_CTX_LWRIE_SIZE      256
#define LW_GR3D_VPE_AR_INDEX_BITS       11
#define LW_GR3D_VPE_OFF_REG_WIDTH       44
#define LW_GR3D_VPE_ISSUE_WINDOW_SIZE   2
#define LW_GR3D_VPE_OBUF_NOP    31
#define VPE_4X_INSTR_WIDTH      127
#define VPE_4X_INSTR_CTX_ADDR_WIDTH     10
#define VPE_4X_INSTR_RT_ADDR_WIDTH      6
#define VPE_4X_INSTR_OPCODE_V_WIDTH     5
#define VPE_4X_INSTR_OPCODE_S_WIDTH     5
#define VPE_4X_INSTR_SEL_SCALAR 0
#define VPE_4X_INSTR_LAST_LSB   0
#define VPE_4X_INSTR_RC_MSB     28
#define VPE_4X_INSTR_RC_LSB     23
#define LW_GR3D_VPE_ATTR_BITS   4
#define LW_GR3D_VPE_STACK_SIZE  8
#define LW_GR3D_VPE_STACK_ADDR_BITS     3
#define LW_GR3D_VPE_RANKINE_STACK_SIZE  4
#define LW_GR3D_VPE_READ_ADDR_BITS      9
#define LW_GR3D_VPE_READ_DATA_WIDTH     128

//
// REGISTER LIST
//
#define LIST_LWE397_REGS(_op_) \
_op_(LWE397_CTL_INCR_SYNCPT_0) \
_op_(LWE397_CTL_INCR_SYNCPT_CNTRL_0) \
_op_(LWE397_CTL_INCR_SYNCPT_ERROR_0) \
_op_(LWE397_CTL_INTSTATUS_0) \
_op_(LWE397_CTL_INTENABLE_0) \
_op_(LWE397_CTL_CTXSW_0) \
_op_(LWE397_CTL_STAT_0) \
_op_(LWE397_CTL_STAT) \
_op_(LWE397_CTL_STAT_1) \
_op_(LWE397_CTL_STAT_CLK_COUNT_0) \
_op_(LWE397_CTL_STAT_CLK_COUNT) \
_op_(LWE397_CTL_STAT_CLK_COUNT_1) \
_op_(LWE397_CTL_STAT_XFER_COUNT_0) \
_op_(LWE397_CTL_STAT_XFER_COUNT) \
_op_(LWE397_CTL_STAT_XFER_COUNT_1) \
_op_(LWE397_CTL_STAT_WAIT_COUNT_0) \
_op_(LWE397_CTL_STAT_WAIT_COUNT) \
_op_(LWE397_CTL_STAT_WAIT_COUNT_1) \
_op_(LWE397_CTL_STAT_EN_COUNT_0) \
_op_(LWE397_CTL_STAT_EN_COUNT) \
_op_(LWE397_CTL_STAT_EN_COUNT_1) \
_op_(LWE397_IDX_ATTRIBUTE_0) \
_op_(LWE397_IDX_ATTRIBUTE) \
_op_(LWE397_IDX_ATTRIBUTE_1) \
_op_(LWE397_IDX_ATTRIBUTE_2) \
_op_(LWE397_IDX_ATTRIBUTE_3) \
_op_(LWE397_IDX_ATTRIBUTE_4) \
_op_(LWE397_IDX_ATTRIBUTE_5) \
_op_(LWE397_IDX_ATTRIBUTE_6) \
_op_(LWE397_IDX_ATTRIBUTE_7) \
_op_(LWE397_IDX_ATTRIBUTE_8) \
_op_(LWE397_IDX_ATTRIBUTE_9) \
_op_(LWE397_IDX_ATTRIBUTE_10) \
_op_(LWE397_IDX_ATTRIBUTE_11) \
_op_(LWE397_IDX_ATTRIBUTE_12) \
_op_(LWE397_IDX_ATTRIBUTE_13) \
_op_(LWE397_IDX_ATTRIBUTE_14) \
_op_(LWE397_IDX_ATTRIBUTE_15) \
_op_(LWE397_IDX_ATTRIBUTE_16) \
_op_(LWE397_IDX_ATTRIBUTE_17) \
_op_(LWE397_IDX_ATTRIBUTE_18) \
_op_(LWE397_IDX_ATTRIBUTE_19) \
_op_(LWE397_IDX_ATTRIBUTE_20) \
_op_(LWE397_IDX_ATTRIBUTE_21) \
_op_(LWE397_IDX_ATTRIBUTE_22) \
_op_(LWE397_IDX_ATTRIBUTE_23) \
_op_(LWE397_IDX_ATTRIBUTE_24) \
_op_(LWE397_IDX_ATTRIBUTE_25) \
_op_(LWE397_IDX_ATTRIBUTE_26) \
_op_(LWE397_IDX_ATTRIBUTE_27) \
_op_(LWE397_IDX_ATTRIBUTE_28) \
_op_(LWE397_IDX_ATTRIBUTE_29) \
_op_(LWE397_IDX_ATTRIBUTE_30) \
_op_(LWE397_IDX_ATTRIBUTE_31) \
_op_(LWE397_IDX_ATTR_MASK_0) \
_op_(LWE397_IDX_INDEX_BASE_0) \
_op_(LWE397_IDX_SET_PRIM_0) \
_op_(LWE397_IDX_DRAW_PRIM_0) \
_op_(LWE397_IDX_IDX_CTL_0) \
_op_(LWE397_IDX_IDX_STAT_0) \
_op_(LWE397_IDX_LW_MCCIF_FIFOCTRL_RO_0) \
_op_(LWE397_VPE_MODE_0) \
_op_(LWE397_VPE_TIMEOUT_0) \
_op_(LWE397_VPE_CONST_READ_LIMIT_0) \
_op_(LWE397_VPE_BRANCHBITS_0) \
_op_(LWE397_VPE_START_0) \
_op_(LWE397_VPE_INST_OFFSET_0) \
_op_(LWE397_VPE_INST_DATA_0) \
_op_(LWE397_VPE_CONST_OFFSET_0) \
_op_(LWE397_VPE_CONST_DATA_0) \
_op_(LWE397_VPE_GEOM_STALL_0) \
_op_(LWE397_VPE_VPE_CTRL_0) \
_op_(LWE397_VPE_VPE_DEBUG_0) \
_op_(LWE397_SU_INST_0) \
_op_(LWE397_SU_INST) \
_op_(LWE397_SU_INST_1) \
_op_(LWE397_SU_INST_2) \
_op_(LWE397_SU_INST_3) \
_op_(LWE397_SU_INST_4) \
_op_(LWE397_SU_INST_5) \
_op_(LWE397_SU_INST_6) \
_op_(LWE397_SU_INST_7) \
_op_(LWE397_SU_INST_8) \
_op_(LWE397_SU_INST_9) \
_op_(LWE397_SU_INST_10) \
_op_(LWE397_SU_INST_11) \
_op_(LWE397_SU_INST_12) \
_op_(LWE397_SU_INST_13) \
_op_(LWE397_SU_INST_14) \
_op_(LWE397_SU_INST_15) \
_op_(LWE397_SU_INST_16) \
_op_(LWE397_SU_INST_17) \
_op_(LWE397_SU_INST_18) \
_op_(LWE397_SU_INST_19) \
_op_(LWE397_SU_INST_20) \
_op_(LWE397_SU_INST_21) \
_op_(LWE397_SU_INST_22) \
_op_(LWE397_SU_INST_23) \
_op_(LWE397_SU_INST_24) \
_op_(LWE397_SU_INST_25) \
_op_(LWE397_SU_INST_26) \
_op_(LWE397_SU_INST_27) \
_op_(LWE397_SU_INST_28) \
_op_(LWE397_SU_INST_29) \
_op_(LWE397_SU_INST_30) \
_op_(LWE397_SU_INST_31) \
_op_(LWE397_SU_INST_32) \
_op_(LWE397_SU_INST_33) \
_op_(LWE397_SU_INST_34) \
_op_(LWE397_SU_INST_35) \
_op_(LWE397_SU_INST_36) \
_op_(LWE397_SU_INST_37) \
_op_(LWE397_SU_INST_38) \
_op_(LWE397_SU_INST_39) \
_op_(LWE397_SU_INST_40) \
_op_(LWE397_SU_INST_41) \
_op_(LWE397_SU_INST_42) \
_op_(LWE397_SU_INST_43) \
_op_(LWE397_SU_INST_44) \
_op_(LWE397_SU_INST_45) \
_op_(LWE397_SU_INST_46) \
_op_(LWE397_SU_INST_47) \
_op_(LWE397_SU_INST_48) \
_op_(LWE397_SU_INST_49) \
_op_(LWE397_SU_INST_50) \
_op_(LWE397_SU_INST_51) \
_op_(LWE397_SU_INST_52) \
_op_(LWE397_SU_INST_53) \
_op_(LWE397_SU_INST_54) \
_op_(LWE397_SU_INST_55) \
_op_(LWE397_SU_INST_56) \
_op_(LWE397_SU_INST_57) \
_op_(LWE397_SU_INST_58) \
_op_(LWE397_SU_INST_59) \
_op_(LWE397_SU_INST_60) \
_op_(LWE397_SU_INST_61) \
_op_(LWE397_SU_INST_62) \
_op_(LWE397_SU_INST_63) \
_op_(LWE397_SU_DRAW_POINT_0) \
_op_(LWE397_SU_DRAW_LINE_0) \
_op_(LWE397_SU_DRAW_TRI_0) \
_op_(LWE397_SU_PARAM_0) \
_op_(LWE397_SU_ZBIAS_0) \
_op_(LWE397_SU_ZFACTOR_0) \
_op_(LWE397_SU_POINT_PARAM_0) \
_op_(LWE397_SU_POINT_WIDTH_2_0) \
_op_(LWE397_SU_POINT_MAX_S_0) \
_op_(LWE397_SU_POINT_MAX_T_0) \
_op_(LWE397_SU_POINT_MIN_S_0) \
_op_(LWE397_SU_POINT_MIN_T_0) \
_op_(LWE397_SU_LINE_PARAM_0) \
_op_(LWE397_SU_LINE_WIDTH_2_0) \
_op_(LWE397_SU_LINE_MAX_ATTR_W_0) \
_op_(LWE397_SU_LINE_MIN_ATTR_W_0) \
_op_(LWE397_SU_SCISSOR_X_0) \
_op_(LWE397_SU_SCISSOR_Y_0) \
_op_(LWE397_SU_VIEWPORT_X_0) \
_op_(LWE397_SU_VIEWPORT_Y_0) \
_op_(LWE397_SU_VIEWPORT_Z_0) \
_op_(LWE397_SU_VIEWPORT_W_0) \
_op_(LWE397_SU_VIEWPORT_H_0) \
_op_(LWE397_SU_VIEWPORT_D_0) \
_op_(LWE397_SU_GUARDBAND_W_0) \
_op_(LWE397_SU_GUARDBAND_H_0) \
_op_(LWE397_SU_GUARDBAND_D_0) \
_op_(LWE397_SU_UCPLANE_0) \
_op_(LWE397_SU_UCPLANE) \
_op_(LWE397_SU_CLKEN_OVERRIDE_0) \
_op_(LWE397_SU_CLIP_CLKEN_OVERRIDE_0) \
_op_(LWE397_SU_OUTER_SLI_SCISSOR_X_0) \
_op_(LWE397_SU_OUTER_SLI_SCISSOR_Y_0) \
_op_(LWE397_QR_S_TEST_0) \
_op_(LWE397_QR_S_TEST) \
_op_(LWE397_QR_S_TEST_1) \
_op_(LWE397_QR_S_CTRL_0) \
_op_(LWE397_QR_Z_TEST_0) \
_op_(LWE397_QR_Z_MIN_0) \
_op_(LWE397_QR_Z_MAX_0) \
_op_(LWE397_QR_RAST_OPERATION_0) \
_op_(LWE397_QR_RAST_SCISSOR_SNAP_0) \
_op_(LWE397_QR_RAST_SCISSOR_MIN_0) \
_op_(LWE397_QR_RAST_SCISSOR_MAX_0) \
_op_(LWE397_QR_RAST_BBOX_MIN_0) \
_op_(LWE397_QR_RAST_BBOX_MAX_0) \
_op_(LWE397_QR_SB_OPERATION_0) \
_op_(LWE397_QR_QRAST_CLKEN_OVERRIDE_0) \
_op_(LWE397_QR_VCAA_OPERATION_0) \
_op_(LWE397_QR_OUTPUT_TO_SHADER_0) \
_op_(LWE397_QR_QRAST_DEBUG_0) \
_op_(LWE397_QR_QRAST_LIMITS_0) \
_op_(LWE397_QR_PIXEL_COUNT_CTRL_0) \
_op_(LWE397_QR_PIXEL_COUNT_0) \
_op_(LWE397_PSEQ_FLUSH_0) \
_op_(LWE397_PSEQ_CTL_0) \
_op_(LWE397_PSEQ_TIMEOUT_0) \
_op_(LWE397_PSEQ_PC_0) \
_op_(LWE397_PSEQ_COMMAND_0) \
_op_(LWE397_PSEQ_COMMAND) \
_op_(LWE397_PSEQ_COMMAND_1) \
_op_(LWE397_PSEQ_COMMAND_2) \
_op_(LWE397_PSEQ_COMMAND_3) \
_op_(LWE397_PSEQ_COMMAND_4) \
_op_(LWE397_PSEQ_COMMAND_5) \
_op_(LWE397_PSEQ_COMMAND_6) \
_op_(LWE397_PSEQ_COMMAND_7) \
_op_(LWE397_PSEQ_COMMAND_8) \
_op_(LWE397_PSEQ_COMMAND_9) \
_op_(LWE397_PSEQ_COMMAND_10) \
_op_(LWE397_PSEQ_COMMAND_11) \
_op_(LWE397_PSEQ_COMMAND_12) \
_op_(LWE397_PSEQ_COMMAND_13) \
_op_(LWE397_PSEQ_COMMAND_14) \
_op_(LWE397_PSEQ_COMMAND_15) \
_op_(LWE397_PSEQ_COMMAND_16) \
_op_(LWE397_PSEQ_COMMAND_17) \
_op_(LWE397_PSEQ_COMMAND_18) \
_op_(LWE397_PSEQ_COMMAND_19) \
_op_(LWE397_PSEQ_COMMAND_20) \
_op_(LWE397_PSEQ_COMMAND_21) \
_op_(LWE397_PSEQ_COMMAND_22) \
_op_(LWE397_PSEQ_COMMAND_23) \
_op_(LWE397_PSEQ_COMMAND_24) \
_op_(LWE397_PSEQ_COMMAND_25) \
_op_(LWE397_PSEQ_COMMAND_26) \
_op_(LWE397_PSEQ_COMMAND_27) \
_op_(LWE397_PSEQ_COMMAND_28) \
_op_(LWE397_PSEQ_COMMAND_29) \
_op_(LWE397_PSEQ_COMMAND_30) \
_op_(LWE397_PSEQ_COMMAND_31) \
_op_(LWE397_PSEQ_INST_OFFSET_0) \
_op_(LWE397_PSEQ_INST_DATA_0) \
_op_(LWE397_PSEQ_DBG_X_0) \
_op_(LWE397_PSEQ_DBG_Y_0) \
_op_(LWE397_PSEQ_DBG_CTL_0) \
_op_(LWE397_PSEQ_QUAD_ID_0) \
_op_(LWE397_PSEQ_DWR_IF_STATE_0) \
_op_(LWE397_AT_REMAP_OFFSET_0) \
_op_(LWE397_AT_REMAP_DATA_0) \
_op_(LWE397_AT_REMAP_DATA_4X_0) \
_op_(LWE397_AT_INST_OFFSET_0) \
_op_(LWE397_AT_INST_DATA_0) \
_op_(LWE397_AT_CONSTANT0_0) \
_op_(LWE397_AT_CONSTANT0) \
_op_(LWE397_AT_CONSTANT0_1) \
_op_(LWE397_AT_CONSTANT0_2) \
_op_(LWE397_AT_CONSTANT0_3) \
_op_(LWE397_AT_TRAM_OFFSET_0) \
_op_(LWE397_AT_TRAM_DATA_0) \
_op_(LWE397_AT_CLKEN_OVERRIDE_0) \
_op_(LWE397_TEX_INST_OFFSET_0) \
_op_(LWE397_TEX_INST_DATA_0) \
_op_(LWE397_TEX_COLORKEY_0) \
_op_(LWE397_TEX_TEXADDR_0) \
_op_(LWE397_TEX_TEXADDR) \
_op_(LWE397_TEX_TEXADDR_1) \
_op_(LWE397_TEX_TEXADDR_2) \
_op_(LWE397_TEX_TEXADDR_3) \
_op_(LWE397_TEX_TEXADDR_4) \
_op_(LWE397_TEX_TEXADDR_5) \
_op_(LWE397_TEX_TEXADDR_6) \
_op_(LWE397_TEX_TEXADDR_7) \
_op_(LWE397_TEX_TEXADDR_8) \
_op_(LWE397_TEX_TEXADDR_9) \
_op_(LWE397_TEX_TEXADDR_10) \
_op_(LWE397_TEX_TEXADDR_11) \
_op_(LWE397_TEX_TEXADDR_12) \
_op_(LWE397_TEX_TEXADDR_13) \
_op_(LWE397_TEX_TEXADDR_14) \
_op_(LWE397_TEX_TEXADDR_15) \
_op_(LWE397_TEX_TEXDESC_0) \
_op_(LWE397_TEX_TEXDESC) \
_op_(LWE397_TEX_TEXDESC_1) \
_op_(LWE397_TEX_TEXDESC_2) \
_op_(LWE397_TEX_TEXDESC_3) \
_op_(LWE397_TEX_TEXDESC_4) \
_op_(LWE397_TEX_TEXDESC_5) \
_op_(LWE397_TEX_TEXDESC_6) \
_op_(LWE397_TEX_TEXDESC_7) \
_op_(LWE397_TEX_TEXDESC_8) \
_op_(LWE397_TEX_TEXDESC_9) \
_op_(LWE397_TEX_TEXDESC_10) \
_op_(LWE397_TEX_TEXDESC_11) \
_op_(LWE397_TEX_TEXDESC_12) \
_op_(LWE397_TEX_TEXDESC_13) \
_op_(LWE397_TEX_TEXDESC_14) \
_op_(LWE397_TEX_TEXDESC_15) \
_op_(LWE397_TEX_TEXDESC_16) \
_op_(LWE397_TEX_TEXDESC_17) \
_op_(LWE397_TEX_TEXDESC_18) \
_op_(LWE397_TEX_TEXDESC_19) \
_op_(LWE397_TEX_TEXDESC_20) \
_op_(LWE397_TEX_TEXDESC_21) \
_op_(LWE397_TEX_TEXDESC_22) \
_op_(LWE397_TEX_TEXDESC_23) \
_op_(LWE397_TEX_TEXDESC_24) \
_op_(LWE397_TEX_TEXDESC_25) \
_op_(LWE397_TEX_TEXDESC_26) \
_op_(LWE397_TEX_TEXDESC_27) \
_op_(LWE397_TEX_TEXDESC_28) \
_op_(LWE397_TEX_TEXDESC_29) \
_op_(LWE397_TEX_TEXDESC_30) \
_op_(LWE397_TEX_TEXDESC_31) \
_op_(LWE397_TEX_TEXCTL_0) \
_op_(LWE397_TEX_CLKEN_OVERRIDE_0) \
_op_(LWE397_TEX_LW_MCCIF_FIFOCTRL_RO_0) \
_op_(LWE397_TEX_TEXDESC_NPOT_AUX_0) \
_op_(LWE397_TEX_TEXDESC_NPOT_AUX) \
_op_(LWE397_TEX_TEXDESC_NPOT_AUX_1) \
_op_(LWE397_TEX_TEXDESC_NPOT_AUX_2) \
_op_(LWE397_TEX_TEXDESC_NPOT_AUX_3) \
_op_(LWE397_TEX_TEXDESC_NPOT_AUX_4) \
_op_(LWE397_TEX_TEXDESC_NPOT_AUX_5) \
_op_(LWE397_TEX_TEXDESC_NPOT_AUX_6) \
_op_(LWE397_TEX_TEXDESC_NPOT_AUX_7) \
_op_(LWE397_TEX_TEXDESC_NPOT_AUX_8) \
_op_(LWE397_TEX_TEXDESC_NPOT_AUX_9) \
_op_(LWE397_TEX_TEXDESC_NPOT_AUX_10) \
_op_(LWE397_TEX_TEXDESC_NPOT_AUX_11) \
_op_(LWE397_TEX_TEXDESC_NPOT_AUX_12) \
_op_(LWE397_TEX_TEXDESC_NPOT_AUX_13) \
_op_(LWE397_TEX_TEXDESC_NPOT_AUX_14) \
_op_(LWE397_TEX_TEXDESC_NPOT_AUX_15) \
_op_(LWE397_ALU_REMAP_OFFSET_0) \
_op_(LWE397_ALU_REMAP_DATA_0) \
_op_(LWE397_ALU_REMAP_DATA_4X_0) \
_op_(LWE397_ALU_INST_OFFSET_0) \
_op_(LWE397_ALU_INST_DATA_0) \
_op_(LWE397_ALU_P2CX_OFFSET_0) \
_op_(LWE397_ALU_P2CX_DATA_0) \
_op_(LWE397_ALU_GLOBALS_0) \
_op_(LWE397_ALU_GLOBALS) \
_op_(LWE397_ALU_GLOBALS_1) \
_op_(LWE397_ALU_GLOBALS_2) \
_op_(LWE397_ALU_GLOBALS_3) \
_op_(LWE397_ALU_GLOBALS_4) \
_op_(LWE397_ALU_GLOBALS_5) \
_op_(LWE397_ALU_GLOBALS_6) \
_op_(LWE397_ALU_GLOBALS_7) \
_op_(LWE397_ALU_GLOBALS_8) \
_op_(LWE397_ALU_GLOBALS_9) \
_op_(LWE397_ALU_GLOBALS_10) \
_op_(LWE397_ALU_GLOBALS_11) \
_op_(LWE397_ALU_GLOBALS_12) \
_op_(LWE397_ALU_GLOBALS_13) \
_op_(LWE397_ALU_GLOBALS_14) \
_op_(LWE397_ALU_GLOBALS_15) \
_op_(LWE397_ALU_GLOBALS_16) \
_op_(LWE397_ALU_GLOBALS_17) \
_op_(LWE397_ALU_GLOBALS_18) \
_op_(LWE397_ALU_GLOBALS_19) \
_op_(LWE397_ALU_GLOBALS_20) \
_op_(LWE397_ALU_GLOBALS_21) \
_op_(LWE397_ALU_GLOBALS_22) \
_op_(LWE397_ALU_GLOBALS_23) \
_op_(LWE397_ALU_GLOBALS_24) \
_op_(LWE397_ALU_GLOBALS_25) \
_op_(LWE397_ALU_GLOBALS_26) \
_op_(LWE397_ALU_GLOBALS_27) \
_op_(LWE397_ALU_GLOBALS_28) \
_op_(LWE397_ALU_GLOBALS_29) \
_op_(LWE397_ALU_GLOBALS_30) \
_op_(LWE397_ALU_GLOBALS_31) \
_op_(LWE397_DW_INST_OFFSET_0) \
_op_(LWE397_DW_INST_DATA_0) \
_op_(LWE397_DW_LOGIC_OP_0) \
_op_(LWE397_DW_ST_ENABLE_0) \
_op_(LWE397_DW_MEMORY_OUTPUT_ADDRESS_0) \
_op_(LWE397_DW_MEMORY_OUTPUT_DATA_0) \
_op_(LWE397_DW_MEMORY_OUTPUT_INCR_0) \
_op_(LWE397_DW_TIMESTAMP_CTL_0) \
_op_(LWE397_DW_TIMESTAMP_LOW_0) \
_op_(LWE397_DW_TIMESTAMP_HIGH_0) \
_op_(LWE397_DW_PIXEL_COUNT_CTRL_0) \
_op_(LWE397_DW_PIXEL_COUNT_0) \
_op_(LWE397_FDC_CONTROL_0) \
_op_(LWE397_FDC_STATUS_0) \
_op_(LWE397_FDC_MAX_QZ_LINES_0) \
_op_(LWE397_FDC_MAX_QV_LINES_0) \
_op_(LWE397_FDC_MAX_QS_LINES_0) \
_op_(LWE397_FDC_MAX_PS_LINES_0) \
_op_(LWE397_FDC_MAX_Q_LINES_0) \
_op_(LWE397_FDC_MAX_Q_P_LINES_0) \
_op_(LWE397_FDC_FLUSH_CTL_0) \
_op_(LWE397_FDC_L1_TIMEOUT_0) \
_op_(LWE397_FDC_INSTRUMENT_0) \
_op_(LWE397_FDC_CLKEN_OVERRIDE_0) \
_op_(LWE397_FDC_LW_MCCIF_FIFOCTRL_RO_0) \
_op_(LWE397_GSHIM_WRITE_MASK_0) \
_op_(LWE397_GSHIM_READ_SELECT_0) \
_op_(LWE397_GSHIM_FLUSH_0) \
_op_(LWE397_GSHIM_SCISSOR_SPLIT_0) \
_op_(LWE397_GSHIM_STAT_ENABLE_0) \
_op_(LWE397_GSHIM_STAT_STALL_0) \
_op_(LWE397_GSHIM_STAT_STALL) \
_op_(LWE397_GSHIM_STAT_STALL_1) \
_op_(LWE397_GSHIM_STAT_WAIT_0) \
_op_(LWE397_GSHIM_STAT_WAIT) \
_op_(LWE397_GSHIM_STAT_WAIT_1) \
_op_(LWE397_GSHIM_STAT_COMB_0) \
_op_(LWE397_GSHIM_STAT_COMB) \
_op_(LWE397_GSHIM_STAT_COMB_1) \
_op_(LWE397_GSHIM_STAT_HWR_WAIT_0) \
_op_(LWE397_GSHIM_STAT_HWR_XFER_0) \
_op_(LWE397_GSHIM_STAT_SYNCPT_WAIT_0) \
_op_(LWE397_GSHIM_STAT_SYNCPT_WAIT) \
_op_(LWE397_GSHIM_STAT_SYNCPT_WAIT_1) \
_op_(LWE397_GSHIM_DLB_CONTROL_0) \
_op_(LWE397_GSHIM_DLB_0) \
_op_(LWE397_GSHIM_DLB_TRIGGER_0) \
_op_(LWE397_GSHIM_HIGH_WATERMARK_0) \
_op_(LWE397_GSHIM_DEBUG0_0) \
_op_(LWE397_GLOBAL_SURFADDR_0) \
_op_(LWE397_GLOBAL_SURFADDR) \
_op_(LWE397_GLOBAL_SURFADDR_1) \
_op_(LWE397_GLOBAL_SURFADDR_2) \
_op_(LWE397_GLOBAL_SURFADDR_3) \
_op_(LWE397_GLOBAL_SURFADDR_4) \
_op_(LWE397_GLOBAL_SURFADDR_5) \
_op_(LWE397_GLOBAL_SURFADDR_6) \
_op_(LWE397_GLOBAL_SURFADDR_7) \
_op_(LWE397_GLOBAL_SURFADDR_8) \
_op_(LWE397_GLOBAL_SURFADDR_9) \
_op_(LWE397_GLOBAL_SURFADDR_10) \
_op_(LWE397_GLOBAL_SURFADDR_11) \
_op_(LWE397_GLOBAL_SURFADDR_12) \
_op_(LWE397_GLOBAL_SURFADDR_13) \
_op_(LWE397_GLOBAL_SURFADDR_14) \
_op_(LWE397_GLOBAL_SURFADDR_15) \
_op_(LWE397_GLOBAL_SURFDESC_0) \
_op_(LWE397_GLOBAL_SURFDESC) \
_op_(LWE397_GLOBAL_SURFDESC_1) \
_op_(LWE397_GLOBAL_SURFDESC_2) \
_op_(LWE397_GLOBAL_SURFDESC_3) \
_op_(LWE397_GLOBAL_SURFDESC_4) \
_op_(LWE397_GLOBAL_SURFDESC_5) \
_op_(LWE397_GLOBAL_SURFDESC_6) \
_op_(LWE397_GLOBAL_SURFDESC_7) \
_op_(LWE397_GLOBAL_SURFDESC_8) \
_op_(LWE397_GLOBAL_SURFDESC_9) \
_op_(LWE397_GLOBAL_SURFDESC_10) \
_op_(LWE397_GLOBAL_SURFDESC_11) \
_op_(LWE397_GLOBAL_SURFDESC_12) \
_op_(LWE397_GLOBAL_SURFDESC_13) \
_op_(LWE397_GLOBAL_SURFDESC_14) \
_op_(LWE397_GLOBAL_SURFDESC_15) \
_op_(LWE397_GLOBAL_PIX_ATTR_0) \
_op_(LWE397_GLOBAL_TRI_ATTR_0) \
_op_(LWE397_GLOBAL_INST_OFFSET_0) \
_op_(LWE397_GLOBAL_RAISE_0) \
_op_(LWE397_GLOBAL_REFCNT_0) \
_op_(LWE397_GLOBAL_INSTRUMENT_0) \
_op_(LWE397_GLOBAL_DITHER_TABLE_0) \
_op_(LWE397_GLOBAL_FLUSH_0) \
_op_(LWE397_GLOBAL_S_OPERATION_0) \
_op_(LWE397_GLOBAL_S_OPERATION) \
_op_(LWE397_GLOBAL_S_OPERATION_1) \
_op_(LWE397_GLOBAL_SPILLSURFADDR_0) \
_op_(LWE397_GLOBAL_LW_MCCIF_FIFOCTRL_0) \
_op_(LWE397_GLOBAL_SURFOVERADDR_0) \
_op_(LWE397_GLOBAL_SURFOVERADDR) \
_op_(LWE397_GLOBAL_SURFOVERADDR_1) \
_op_(LWE397_GLOBAL_SURFOVERADDR_2) \
_op_(LWE397_GLOBAL_SURFOVERADDR_3) \
_op_(LWE397_GLOBAL_SURFOVERADDR_4) \
_op_(LWE397_GLOBAL_SURFOVERADDR_5) \
_op_(LWE397_GLOBAL_SURFOVERADDR_6) \
_op_(LWE397_GLOBAL_SURFOVERADDR_7) \
_op_(LWE397_GLOBAL_SURFOVERADDR_8) \
_op_(LWE397_GLOBAL_SURFOVERADDR_9) \
_op_(LWE397_GLOBAL_SURFOVERADDR_10) \
_op_(LWE397_GLOBAL_SURFOVERADDR_11) \
_op_(LWE397_GLOBAL_SURFOVERADDR_12) \
_op_(LWE397_GLOBAL_SURFOVERADDR_13) \
_op_(LWE397_GLOBAL_SURFOVERADDR_14) \
_op_(LWE397_GLOBAL_SURFOVERADDR_15) \
_op_(LWE397_GLOBAL_MEMORY_OUTPUT_READS_0) \
_op_(LWE397_GLOBAL_HORIZONTAL_SWATH_RENDERING_0) \
_op_(LWE397_GLOBAL_INNER_SLI_SCISSOR_X_0) \
_op_(LWE397_GLOBAL_INNER_SLI_SCISSOR_Y_0)


//
// ADDRESS SPACES
//
#define BASE_ADDRESS_LWE397_CTL   0x00000000
#define BASE_ADDRESS_LWE397_IDX   0x00000100
#define BASE_ADDRESS_LWE397_VPE   0x00000200
#define BASE_ADDRESS_LWE397_SU    0x00000300
#define BASE_ADDRESS_LWE397_QR    0x00000400
#define BASE_ADDRESS_LWE397_PSEQ  0x00000500
#define BASE_ADDRESS_LWE397_AT    0x00000600
#define BASE_ADDRESS_LWE397_TEX   0x00000700
#define BASE_ADDRESS_LWE397_ALU   0x00000800
#define BASE_ADDRESS_LWE397_DW    0x00000900
#define BASE_ADDRESS_LWE397_FDC   0x00000a00
#define BASE_ADDRESS_LWE397_GSHIM 0x00000b00
#define BASE_ADDRESS_LWE397_GLOBAL        0x00000e00

//
// AR3D REGISTER BANKS
//
#define LWE397_CTL0_FIRST_REG 0x0000 // LWE397_CTL_INCR_SYNCPT_0
#define LWE397_CTL0_LAST_REG 0x0002 // LWE397_CTL_INCR_SYNCPT_ERROR_0
#define LWE397_CTL1_FIRST_REG 0x0008 // LWE397_CTL_INTSTATUS_0
#define LWE397_CTL1_LAST_REG 0x000a // LWE397_CTL_CTXSW_0
#define LWE397_CTL2_FIRST_REG 0x000c // LWE397_CTL_STAT_0
#define LWE397_CTL2_LAST_REG 0x0015 // LWE397_CTL_STAT_EN_COUNT_1
#define LWE397_IDX0_FIRST_REG 0x0100 // LWE397_IDX_ATTRIBUTE_0
#define LWE397_IDX0_LAST_REG 0x0126 // LWE397_IDX_LW_MCCIF_FIFOCTRL_RO_0
#define LWE397_VPE0_FIRST_REG 0x0200 // LWE397_VPE_MODE_0
#define LWE397_VPE0_LAST_REG 0x020b // LWE397_VPE_VPE_DEBUG_0
#define LWE397_SU0_FIRST_REG 0x0300 // LWE397_SU_INST_0
#define LWE397_SU0_LAST_REG 0x035b // LWE397_SU_UCPLANE_0
#define LWE397_SU1_FIRST_REG 0x0363 // LWE397_SU_CLKEN_OVERRIDE_0
#define LWE397_SU1_LAST_REG 0x0366 // LWE397_SU_OUTER_SLI_SCISSOR_Y_0
#define LWE397_QR0_FIRST_REG 0x0400 // LWE397_QR_S_TEST_0
#define LWE397_QR0_LAST_REG 0x0413 // LWE397_QR_PIXEL_COUNT_0
#define LWE397_PSEQ0_FIRST_REG 0x0500 // LWE397_PSEQ_FLUSH_0
#define LWE397_PSEQ0_LAST_REG 0x0503 // LWE397_PSEQ_PC_0
#define LWE397_PSEQ1_FIRST_REG 0x0520 // LWE397_PSEQ_COMMAND_0
#define LWE397_PSEQ1_LAST_REG 0x0546 // LWE397_PSEQ_DWR_IF_STATE_0
#define LWE397_AT0_FIRST_REG 0x0600 // LWE397_AT_REMAP_OFFSET_0
#define LWE397_AT0_LAST_REG 0x0604 // LWE397_AT_INST_DATA_0
#define LWE397_AT1_FIRST_REG 0x0608 // LWE397_AT_CONSTANT0_0
#define LWE397_AT1_LAST_REG 0x060e // LWE397_AT_CLKEN_OVERRIDE_0
#define LWE397_TEX0_FIRST_REG 0x0700 // LWE397_TEX_INST_OFFSET_0
#define LWE397_TEX0_LAST_REG 0x0702 // LWE397_TEX_COLORKEY_0
#define LWE397_TEX1_FIRST_REG 0x0710 // LWE397_TEX_TEXADDR_0
#define LWE397_TEX1_LAST_REG 0x0742 // LWE397_TEX_LW_MCCIF_FIFOCTRL_RO_0
#define LWE397_TEX2_FIRST_REG 0x0750 // LWE397_TEX_TEXDESC_NPOT_AUX_0
#define LWE397_TEX2_LAST_REG 0x075f // LWE397_TEX_TEXDESC_NPOT_AUX_15
#define LWE397_ALU0_FIRST_REG 0x0800 // LWE397_ALU_REMAP_OFFSET_0
#define LWE397_ALU0_LAST_REG 0x0806 // LWE397_ALU_P2CX_DATA_0
#define LWE397_ALU1_FIRST_REG 0x0820 // LWE397_ALU_GLOBALS_0
#define LWE397_ALU1_LAST_REG 0x083f // LWE397_ALU_GLOBALS_31
#define LWE397_DW0_FIRST_REG 0x0900 // LWE397_DW_INST_OFFSET_0
#define LWE397_DW0_LAST_REG 0x090b // LWE397_DW_PIXEL_COUNT_0
#define LWE397_FDC0_FIRST_REG 0x0a00 // LWE397_FDC_CONTROL_0
#define LWE397_FDC0_LAST_REG 0x0a0c // LWE397_FDC_LW_MCCIF_FIFOCTRL_RO_0
#define LWE397_GSHIM0_FIRST_REG 0x0b00 // LWE397_GSHIM_WRITE_MASK_0
#define LWE397_GSHIM0_LAST_REG 0x0b04 // LWE397_GSHIM_STAT_ENABLE_0
#define LWE397_GSHIM1_FIRST_REG 0x0b06 // LWE397_GSHIM_STAT_STALL_0
#define LWE397_GSHIM1_LAST_REG 0x0b14 // LWE397_GSHIM_DEBUG0_0
#define LWE397_GLOBAL0_FIRST_REG 0x0e00 // LWE397_GLOBAL_SURFADDR_0
#define LWE397_GLOBAL0_LAST_REG 0x0e2b // LWE397_GLOBAL_LW_MCCIF_FIFOCTRL_0
#define LWE397_GLOBAL1_FIRST_REG 0x0e30 // LWE397_GLOBAL_SURFOVERADDR_0
#define LWE397_GLOBAL1_LAST_REG 0x0e43 // LWE397_GLOBAL_INNER_SLI_SCISSOR_Y_0
*/

#endif // ifndef _e397_h_

