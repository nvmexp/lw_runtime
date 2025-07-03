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
#ifndef _cl_e2_twod_h_
#define _cl_e2_twod_h_

#include "lwtypes.h"
#define LW_E2_TWOD                             0xE22D
#define LWE22D_REG_OFFSET_CTX_SIZE             4096

/*
 * The client should specify the class ctx requested in LW_GR_ALLOCATION_PARAMETERS.caps 
 * during alloc.  For example set LW_GR_ALLOCATION_PARAMETERS.caps = LWE22D_SB_CLASS_ID to 
 * reserve the LW_GRAPHICS_2D_SB_CLASS_ID ctx. 
 *  
 * There are 8 2D contexts but they are not all equivalent. 2D engine has several categories
 * of operations it supports: BitBLT, Alpha Blend, Fast Rotate (FR), Line Draw, StretchBLT (SB),
 * VCAA. Each context only supports a subset of these: 
 *  
 *  Name  Class/Context IDs  Capabilities
 *  G2    50,53,56           BitBLT except mono expansion, FR
 *  G2+   51,54              BitBLT including mono expansion, FR, Line Draw, VCAA
 *  SB    52,55,57           StretchBLT
 */
#define LWE22D_DOWNLOAD_CLASS_ID        0x50
#define LWE22D_CLASS_ID                 0x51
#define LWE22D_SB_CLASS_ID              0x52
#define LWE22D_DOWNLOAD_CTX1_CLASS_ID   0x54
#define LWE22D_CTX1_CLASS_ID            0x55
#define LWE22D_SB_CTX1_CLASS_ID         0x56
#define LWE22D_DOWNLOAD_CTX2_CLASS_ID   0x58
#define LWE22D_SB_CTX2_CLASS_ID         0x5A

typedef volatile struct _cle22d_tag0 {
    LwU32 LWE22D_INCR_SYNCPT_0;
    LwU32 LWE22D_INCR_SYNCPT_CNTRL_0;
    LwU32 LWE22D_INCR_SYNCPT_ERROR_0;
    LwU32 LWE22D_G2CLASSCHANNEL_REGONLY_0;
    LwU32 LWE22D_G2TRIGGER_0;
    LwU32 LWE22D_G2TRIGGER1_0;
    LwU32 LWE22D_G2TRIGGER2_0;
    LwU32 LWE22D_G2CMDSEL_0;
    LwU32 LWE22D_G2RAISE_0;
    LwU32 LWE22D_G2HOSTSET_0;
    LwU32 LWE22D_G2HOSTFIFO_0;
    LwU32 LWE22D_G2VDDA_0;
    LwU32 LWE22D_G2VDDAINI_0;
    LwU32 LWE22D_G2HDDA_0;
    LwU32 LWE22D_G2HDDAINILS_0;
    LwU32 LWE22D_G2CSCFIRST_0;
    LwU32 LWE22D_G2CSCSECOND_0;
    LwU32 LWE22D_G2CSCTHIRD_0;
    LwU32 LWE22D_G2CMKEYL_0;
    LwU32 LWE22D_G2CMKEYU_0;
    LwU32 LWE22D_G2UBA_A_0;
    LwU32 LWE22D_G2VBA_A_0;
    LwU32 LWE22D_G2SBFORMAT_0;
    LwU32 LWE22D_G2CONTROLSB_0;
    LwU32 LWE22D_G2CONTROLSECOND_0;
    LwU32 LWE22D_G2CONTROLMAIN_0;
    LwU32 LWE22D_G2ROPFADE_0;
    LwU32 LWE22D_G2ALPHABLEND_0;
    LwU32 LWE22D_G2CLIPLEFTTOP_0;
    LwU32 LWE22D_G2CLIPRIGHTBOT_0;
    LwU32 LWE22D_G2PATPACK_0;
    LwU32 LWE22D_G2PATPACK_SIZE_0;
    LwU32 LWE22D_G2PATBA_0;
    LwU32 LWE22D_G2PATOS_0;
    LwU32 LWE22D_G2PATBGC_0;
    LwU32 LWE22D_G2PATFGC_0;
    LwU32 LWE22D_G2PATKEY_0;
    LwU32 LWE22D_G2DSTBA_0;
    LwU32 LWE22D_G2DSTBA_B_0;
    LwU32 LWE22D_G2DSTBA_C_0;
    LwU32 LWE22D_G2DSTST_0;
    LwU32 LWE22D_G2SRCPACK_0;
    LwU32 LWE22D_G2SRCPACK_SIZE_0;
    LwU32 LWE22D_G2SRCBA_0;
    LwU32 LWE22D_G2SRCBA_B_0;
    LwU32 LWE22D_G2SRCST_0;
    LwU32 LWE22D_G2SRCBGC_0;
    LwU32 LWE22D_G2SRCFGC_0;
    LwU32 LWE22D_G2SRCKEY_0;
    LwU32 LWE22D_G2SRCSIZE_0;
    LwU32 LWE22D_G2DSTSIZE_0;
    LwU32 LWE22D_G2SRCPS_0;
    LwU32 LWE22D_G2DSTPS_0;
    LwU32 LWE22D_G2CBDES_0;
    LwU32 LWE22D_G2CBSTRIDE_0;
    LwU32 LWE22D_G2LINESETTING_0;
    LwU32 LWE22D_G2LINEDELTAN_0;
    LwU32 LWE22D_G2LINEDELTAM_0;
    LwU32 LWE22D_G2LINEPOS_0;
    LwU32 LWE22D_G2LINELEN_0;
    LwU32 LWE22D_G2CSCFOURTH_0;
    LwU32 LWE22D_G2SRCST_B_0;
    LwU32 LWE22D_G2UVSTRIDE_0;
    LwU32 LWE22D_G2CBDES2_0;
    LwU32 LWE22D_G2TILEMODE_0;
    LwU32 LWE22D_G2PATBASE_0;
    LwU32 LWE22D_G2SRCBA_SB_SURFBASE_0;
    LwU32 LWE22D_G2DSTBA_SB_SURFBASE_0;
    LwU32 LWE22D_G2DSTBA_B_SB_SURFBASE_0;
    LwU32 LWE22D_G2VBA_A_SB_SURFBASE_0;
    LwU32 LWE22D_G2UBA_A_SB_SURFBASE_0;
} e2_twod_t;



/*
 * 2D syncpt behavior
 *     - Each 2D context has its own syncpt client.
 *     - OP_DONE increments when 2D's memory write client is idle, meaning all 
 *           preceding operations are complete.  
 *     - RD_DONE behaves like OP_DONE
 *     - REG_WR_SAFE behaves like IMMEDIATE meaning a syncpt increment returns
 *         immediately and does not indicate that a register write will NOT block.  
 *         The implications are that 2D's host FIFO can fill and block access to 2D, 
 *         which will indeterminately block a direct CPU access.
 */
#define LWE22D_REG_OFFSET_CTX_SIZE_LOG2                                  12
#define LWE22D_INCR_SYNCPT_NB_CONDS                                      4
#define LWE22D_CTX1_INCR_SYNCPT_NB_CONDS                                 4
#define LWE22D_CTX2_INCR_SYNCPT_NB_CONDS                                 4
#define LWE22D_CTX3_INCR_SYNCPT_NB_CONDS                                 4
#define LWE22D_CTX4_INCR_SYNCPT_NB_CONDS                                 4
#define LWE22D_CTX5_INCR_SYNCPT_NB_CONDS                                 4
#define LWE22D_CTX6_INCR_SYNCPT_NB_CONDS                                 4
#define LWE22D_CTX7_INCR_SYNCPT_NB_CONDS                                 4


#define LWE22D_INCR_SYNCPT_0                                                (0x0)
#define LWE22D_INCR_SYNCPT_0_COND                                           15:8
#define LWE22D_INCR_SYNCPT_0_COND_IMMEDIATE                                 0x00000000
#define LWE22D_INCR_SYNCPT_0_COND_OP_DONE                                   0x00000001
#define LWE22D_INCR_SYNCPT_0_COND_RD_DONE                                   0x00000002
#define LWE22D_INCR_SYNCPT_0_COND_REG_WR_SAFE                               0x00000003
#define LWE22D_INCR_SYNCPT_0_COND_4                                         0x00000004
#define LWE22D_INCR_SYNCPT_0_COND_5                                         0x00000005
#define LWE22D_INCR_SYNCPT_0_COND_6                                         0x00000006
#define LWE22D_INCR_SYNCPT_0_COND_7                                         0x00000007
#define LWE22D_INCR_SYNCPT_0_COND_8                                         0x00000008
#define LWE22D_INCR_SYNCPT_0_COND_9                                         0x00000009
#define LWE22D_INCR_SYNCPT_0_COND_10                                        0x00000010
#define LWE22D_INCR_SYNCPT_0_COND_11                                        0x00000011
#define LWE22D_INCR_SYNCPT_0_COND_12                                        0x00000012
#define LWE22D_INCR_SYNCPT_0_COND_13                                        0x00000013
#define LWE22D_INCR_SYNCPT_0_COND_14                                        0x00000014
#define LWE22D_INCR_SYNCPT_0_COND_15                                        0x00000015
#define LWE22D_INCR_SYNCPT_0_COND_16                                        0x00000016
#define LWE22D_INCR_SYNCPT_0_INDX                                           7:0

/*
 * If NO_STALL is 1, then when fifos are full,
 * INCR_SYNCPT methods will be dropped and the
 * INCR_SYNCPT_ERROR[COND] bit will be set.
 * If NO_STALL is 0, then when fifos are full,
 *
 * If SOFT_RESET is set, then all internal state
 * of the client syncpt block will be reset.
 * To do soft reset, first set SOFT_RESET of
 * all host1x clients affected, then clear all
 * SOFT_RESETs.
 */
#define LWE22D_INCR_SYNCPT_CNTRL_0                                          (0x1)
#define LWE22D_INCR_SYNCPT_CNTRL_0_NO_STALL                                 8:8
#define LWE22D_INCR_SYNCPT_CNTRL_0_SOFT_RESET                               0:0


#define LWE22D_INCR_SYNCPT_ERROR_0                                          (0x2)
#define LWE22D_INCR_SYNCPT_ERROR_0_STATUS                                   31:0

#define LWE22D_G2CLASSCHANNEL_REGONLY_0                                     (0x8)
#define LWE22D_G2CLASSCHANNEL_REGONLY_0_CTX_VALID                           20:20
#define LWE22D_G2CLASSCHANNEL_REGONLY_0_LWRR_CHANNEL                        19:16

/*
 * Control methods, since they are not programmed often, put them is the beginning
 */
#define LWE22D_G2TRIGGER_0                                                  (0x9)
#define LWE22D_G2TRIGGER_0_TRIGGER                                          15:0

#define LWE22D_G2TRIGGER1_0                                                 (0xa)
#define LWE22D_G2TRIGGER1_0_TRIGGER1                                        15:0

#define LWE22D_G2TRIGGER2_0                                                 (0xb)
#define LWE22D_G2TRIGGER2_0_TRIGGER2                                        15:0

#define LWE22D_G2CMDSEL_0                                                   (0xc)
#define LWE22D_G2CMDSEL_0_PRIORITY                                          28:28
#define LWE22D_G2CMDSEL_0_PRIORITY_LOW                                      0x00000000
#define LWE22D_G2CMDSEL_0_PRIORITY_HIGH                                     0x00000001
#define LWE22D_G2CMDSEL_0_LINKVAL                                           27:25
#define LWE22D_G2CMDSEL_0_LINKEN                                            24:24
#define LWE22D_G2CMDSEL_0_BUFFER_INDEX                                      23:16
#define LWE22D_G2CMDSEL_0_FRAME_END                                         15:15
#define LWE22D_G2CMDSEL_0_FRAME_START                                       14:14
#define LWE22D_G2CMDSEL_0_LINK_WAIT_BUF_END                                 12:12
#define LWE22D_G2CMDSEL_0_CLIP_SOURCE_TOP_BOTTOM                            10:10
#define LWE22D_G2CMDSEL_0_CLIP_SOURCE_TOP_BOTTOM_DISABLE                    0x00000000
#define LWE22D_G2CMDSEL_0_CLIP_SOURCE_TOP_BOTTOM_ENABLE                     0x00000001
#define LWE22D_G2CMDSEL_0_G2OUTPUT                                          9:8
#define LWE22D_G2CMDSEL_0_MEMORY                                            0x00000000
#define LWE22D_G2CMDSEL_0_EPP                                               0x00000001
#define LWE22D_G2CMDSEL_0_RESERVED2                                         0x00000002
#define LWE22D_G2CMDSEL_0_RESERVED3                                         0x00000003
#define LWE22D_G2CMDSEL_0_CBSBDISABLE                                       7:7
#define LWE22D_G2CMDSEL_0_HOSTTRIGGER                                       6:6
#define LWE22D_G2CMDSEL_0_VITRIGGER                                         5:5
#define LWE22D_G2CMDSEL_0_CBENABLE                                          4:4
#define LWE22D_G2CMDSEL_0_SBOR2D                                            0:0
#define LWE22D_G2CMDSEL_0_SBOR2D_G2                                         0x00000000
#define LWE22D_G2CMDSEL_0_SBOR2D_SB                                         0x00000001

#define LWE22D_G2RAISE_0                                                    (0xd)
#define LWE22D_G2RAISE_0_RAISECHANNEL                                       19:16
#define LWE22D_G2RAISE_0_RAISE                                              4:0


#define LWE22D_G2HOSTSET_0                                                  (0xf)
#define LWE22D_G2HOSTSET_0_HSTFPXL                                          7:4
#define LWE22D_G2HOSTSET_0_HSTLNGAP                                         3:0


#define LWE22D_G2HOSTFIFO_0                                                 (0x10)
#define LWE22D_G2HOSTFIFO_0_HOSTFIFODATA                                    31:0


/*
 *  Vertical Scaling DDA
 *  Reset value: xxxx-xxxxh
 *  Vertical DDA Step (Increment) Value (VDSTEP[18:0])
 *   (upper 13 bits should be set to '0')
 *  This parameter specifies the increment value of the DDA used 
 *  for vertical scaling and it is in the form of 6-bit integer and 
 *  12-bit fraction.  This value is detemined by the equation:
 *   (Actual_source_height-1-VDTINI) / (Actual_destination_height-1)
 *   Truncate the rest bits to keep the 12bits fraction. Since we have to meet
 *   (Actual_source_height-1)*1.0 >= (Actual_destination_height - 1)*VDSTEP + VDTINI
 *   Refer to bug 353260 for more detail.
 *  The 6-bit integer allows maximum contraction ratio of 1/64, and 
 *   12-bit fraction assures the maximum limit of aclwmulated error less
 *   than half line-spacing for up to approximately 2000 target lines.
 *  For example, image expansion from 240 lines to 400 lines 
 *    VDSTEP[17:0]  =  19'b00_0000_1001_1001_1010   and
 *  image contraction from 240 lines to 150 lines 
 *     VDSTEP[17:0]  =  19'b00_0001_1001_1001_1010.
 */

#define LWE22D_G2VDDA_0                                                     (0x11)
#define LWE22D_G2VDDA_0_VDSTEP                                              31:0

/*
 *  Vertical Scaling DDA Initial Values
 *  Vertical DDA Initial Value for Top-Field (VDTINI[7:0])
 *  This parameter specifies the fraction part of initial 
 *   value of the DDA used for vertical scaling.
 *  Given a pair of conselwtive source image lines representing 
 *   positions 0.0 and 1.0, a destination line-image will be created 
 *   at any position in [0.0, 1.0), that is specified by the fraction
 *   part of the vertical DDA. The 8-bit initial fraction value 
 *   specifies the position of the first output (destination) line-image
 *   located between the first and the second input (source) line-images, 
 *   respectively at position 0.0 and 1.0.
 *  This parameter provides a way to compensate relative displacement of 
 *   top and bottom fields of a source image. Suppose, 2-to-1 interlaced
 *   original video field images #1 and #2 are in the image buffer, to be 
 *   displayed in an overlay window. Assume that the #1 field image is the
 *   top-field. Using VDTINI = 8'hC0 for the top-field image and 
 *   VDBINI = 8'h40 for the bottom-field image, for example, the two images
 *   from #1 and #2 fields get mapped to identically positioned destination
 */
#define LWE22D_G2VDDAINI_0                                                  (0x12)
#define LWE22D_G2VDDAINI_0_VDBINI                                           15:8
#define LWE22D_G2VDDAINI_0_VDTINI                                           7:0

/*
 *  Horizontal Scaling DDA
 *  Reset value: xxxx-xxxxh
 *  Horizontal DDA Step (Increment) Value (HDSTEP[18:0])
 *   (upper 13 bits should be set to '0')
 *  This parameter specifies the increment value of the DDA used 
 *  for horizontal scaling and it is in the form of 6-bit integer and 
 *  12-bit fraction.  This value is detemined by the equation:
 *   (Actual_source_width-1-HDINI) / (Actual_destination_width-1)
 *   Truncate the rest bits to keep the 12bits fraction.
 *   Refer to bug 353260 for more detail.
 *  The 6-bit integer allows maximum contraction ratio of 1/64, and 
 *   12-bit fraction assures the maximum limit of aclwmulated error less
 *   than half pixel-spacing for up to approximately 2000 destination pixels.
 *  For example, image expansion from 720 pixels to 800 pixels 
 *         HDSTEP[17:0]  =  19'b000_0000_1110_0110_0111   and
 *  image contraction from 720 pixels to 250 pixels
 *   HDSTEP[17:0]  =  19'b000_0010_1110_0001_0101.
 */
#define LWE22D_G2HDDA_0                                                     (0x13)
#define LWE22D_G2HDDA_0_HDSTEP                                              31:0


/*
 *  Horizontal Scaling DDA Initial Value
 *  Horizontal DDA Initial Value (HDINI[7:0])
 *  This parameter specifies the fraction part of initial value 
 *   of the DDA used for horizontal scaling.
 *  Given a group of six conselwtive source pixels that 
 *   the two pixels at the center representing positions 0.0 
 *  and 1.0, a destination pixel will be created at any 
 *   position in [0.0, 1.0), that is specified by the fraction part 
 *  of the horizontal DDA. The 8-bit initial fraction 
 *   value specifies the position of the first output (destination) 
 *  pixel located between the first and the second input (source) 
 *   pixels, respectively at position 0.0 and 1.0.
 *  For horizontal scaling this value may normally be set to 0.
 */
#define LWE22D_G2HDDAINILS_0                                                (0x14)
#define LWE22D_G2HDDAINILS_0_HDINI                                          7:0



/*
 *   Register G2SB_G2CSCFIRST_0  //{sb,sbm,index=15}
 *   The next 3 registers contain 8 parameters used by the YCbCr (YUV) 
 *   to RGB color space colwersion logic.  This logic can also be
 *   used for RGB gain/gamma correction in RGB to RGB modes, and for
 *   luma gain in YUV to YUV modes.  For SC20 two new parameters 
 *   (G2U, G2V) have been added for RGB to YUV colwersion.  These are 
 *   in address 0x38.  The equtions have the form:
 *      YUV->RGB
 *   R = (CYX * (yin + YOS)) + ((LWR * uin) + (CVR * vin))
 *   G = (CYX * (yin + YOS)) + ((LWG * uin) + (CVG * vin))
 *   B = (CYX * (yin + YOS)) + ((LWB * uin) + (CVB * vin))
 *      RGB -> RGB
 *   R = (CVR * rin) + (LWR * bin)
 *   G = (CYX * (gin + YOS)) + ((LWG * bin) + (CVG * rin))
 *   B = (LWB * bin) + (CVB * rin)
 *      YUV -> YUV (not what rtl does: see note below)
 *   Y = (CYX * yin + YOS) + ((LWB * uin) + (CVB * vin))
 *   U = (LWG * uin) + (CVG * vin)
 *   V = (CVR * vin) + (LWR * uin)
 *      RGB -> YUV (new for SC20)
 *   Y = (CVB * rin) + (LWB * bin) + (CYX * gin) + YOS
 *   U = (CVG * rin) + (LWG * bin) + (G2U * gin)
 *   V = (CVR * rin) + (LWR * bin) + (G2V * gin)
 *  
 *   NOTE: YUV->YUV above is definitely wrong
 *   the cmodel *appears* to match the RTL with the following pseudo-code:
 *  
 *  if (yuv2rgb)
 *      (r,g,b) = matrix*(y+sbreg_yos,u,v)
 *      matrix:
 *          sbreg_cyx, sbreg_lwr, sbreg_cvr,
 *          sbreg_cyx, sbreg_lwg, sbreg_cvg,
 *          sbreg_cyx, sbreg_lwb, sbreg_cvb,
 *  else if (rgb2rgb)
 *      (r,g,b) = matrix*(r,g+sbreg_yos,b)
 *      matrix:
 *          sbreg_cvr, 0,         sbreg_lwr, 
 *          sbreg_cvg, sbreg_cyx, sbreg_lwg,
 *          sbreg_cvb, 0,         sbreg_lwb,
 *  else if (yuv2yuv)
 *      (y,u,v) = matrix*(y,u,v)
 *      matrix:
 *          sbreg_lwb, 0,         sbreg_cvb,
 *          sbreg_lwg, sbreg_cyx, sbreg_cvg,
 *          sbreg_lwr, 0,         sbreg_cvr,
 *  else if (rgb2yuv)
 *      (y,u,v) = matrix*(r,g,b) +(sbreg_yos,0,0)
 *      matrix:
 *          sbreg_cvb, sbreg_cyx, sbreg_lwb,
 *          sbreg_cvg, sbreg_g2u, sbreg_lwg,
 *          sbreg_cvr, sbreg_g2v, sbreg_lwr, 

 *   YOS:
 *  Y-Offset (YOFFSET[7:0]) for YUV generation
 *  This parameter consists of 8-bit 2's complement in the range [-128,127].
 *   For YUV->RGB the recommended value is -16 (decimal) or 0xF0
 *   For YUV->YUV &  RGB->RGB, this parameter should be set to 0
 *   For RGB->YUV the recommended value is +16 (decimal) or 0x10
 *   CVR:
 *   multiplier for V/R for V or R generation.
 *  This parameter consists of a sign bit and 9-bit magnitude (s2.7)
 *   For YUV->RGB the recommended value is 1.5960 (decimal) or 0x0CC
 *  If source data is in RGB format, this parameter 
 *   may be used as gain adjustment for R component.
 *   For RGB->YUV the recommended value is +0.439 (decimal) or 0x038
 *  23-22  Reserved
 *  
 *   LWB:
 *   multiplier for U/B for Y or B generation.
 *   consists of a sign bit and 9-bit magnitude (s2.7). 
 *   For YUV->RGB, the recommended value is 2.0172 (decimal) or 0x102.
 *   If source data is in RGB format, this parameter 
 *   may be used as gain adjustment for B component.
 *   For SC20, this register changes precision when doing RGB to YUV 
 *   colwersion (SIFMT=1xxx, DIFMT=0xxx).  LWB becomes s1.8 and the
 *   recommended value is +0.098 or 0x019
 */
#define LWE22D_G2CSCFIRST_0                                                 (0x15)
#define LWE22D_G2CSCFIRST_0_YOS                                             31:24 
#define LWE22D_G2CSCFIRST_0_CVR                                             21:12
#define LWE22D_G2CSCFIRST_0_LWB                                             9:0

/*
 *   CYX:
 *   multiplier for Y/G (G gain)
 *  This positive-only parameter consists of 8-bit magnitude (1.7)
 *   For YUV->YUV the recommended value is 1.1644 (decimal) or 0x95
 *   For YUV->YUV or RGB->RGB this may be used as gain adjustment
 *   for Y or G component.
 *   For RGB->YUV the recommended value is +0.504 (decimal) or 0x041
 *   LWR:
 *   multiplier for U/B for V or R generation.
 *  This parameter consists of a sign bit and 9-bit magnitude (s2.7)
 *    For YUV->RGB normally this parameter is programmed to 0.0.  
 *   This parameter takes non-zero value if hue is rotated. 
 *   For YUV->YUV &  RGB->RGB, this parameter should be set to 0
 *   For RGB->YUV the recommended value is -0.071 (decimal) or 0x209
 *   LWG:
 *   multiplier for U/B for U or G generation.
 *   consists of a sign bit and 8-bit magnitude (s1.7)
 *   For YUV->RGB the recommended value is -0.3918 (decimal) or 0x132
 *   For  RGB->RGB, this parameter should be set to 0
 *   For  YUV->YUV, this parameter should be set to 1 (0x080)
 *   For RGB->YUV the recommended value is +0.439 (decimal) or 0x038
 */
#define LWE22D_G2CSCSECOND_0                                                (0x16)
#define LWE22D_G2CSCSECOND_0_CYX                                            31:24
#define LWE22D_G2CSCSECOND_0_LWR                                            21:12
#define LWE22D_G2CSCSECOND_0_LWG                                            8:0

/*
 *   CVB:
 *   multiplier for V/R for Y or B generation. 
 *  This parameter consists of a sign bit and 9-bit magnitude (s2.7)
 *  For YUV->RGB, normally this parameter is programmed to 0.0.  
 *   This coefficient takes non-zero value if hue is rotated. 
 *   For YUV->YUV &  RGB->RGB, this parameter should be set to 0
 *   For RGB->YUV the recommended value is +0.257 (decimal) or 0x021
 *   CVG:
 *   multiplier for V/R for U or G generation.
 *  This parameter consists of a sign bit and 8-bit magnitude (s1.7)
 *   For YUV->RGB the recommended value is -0.8130 (decimal) or 0x168
 *   For YUV->YUV &  RGB->RGB, this parameter should be set to 0
 *   For RGB->YUV the recommended value is -0.148 (decimal) or 0x113
 */
#define LWE22D_G2CSCTHIRD_0                                                 (0x17)
#define LWE22D_G2CSCTHIRD_0_CVB                                             25:16
#define LWE22D_G2CSCTHIRD_0_CVG                                             8:0

/*
 *   StretchBLT, Color/Chroma Key Lower Limit Register

 *   Register G2SB_G2CMKEYL_0  //{sb,sbm,index=18}
 *   When key generation is enabled, the value of this register is the 
 *   lower color/chroma limit and G2CMKEYU is the upper color/chroma 
 *   limit for component pixel data.
 *   Three component signal values (YCrCb/YUV or RGB) of 
 *   every output pixel are compared to a set of three 
 *   ranges that are specified by three pairs of lower and 
 *   upper color/chroma key values.  If the Key Polarity is 
 *   0, the Key is set to 1 only when all three component values 
 *   of an input pixel are respectively within the set 
 *   of ranges (inclusive of the limit values), else the Key is set to 0.  
 *   If the Key Polarity is 1, the Key is set to 1 
 *   when any of the three component values of an input pixel 
 *   are outside the set of ranges, else the Key is set to 0.
 *  
 *  CKRL:
 *  R or Cr Color Chroma Key Lower Limit (CKRL[7:0])
 *   Cr signal must be treated in offset binary format 
 *   so that the binary interpretation retains monotonicity from 
 *   the minimum signal level to the maximum signal level.
 *  
 *  CKGL:
 *  G or Cb Color Chroma Key Lower Limit (CKGL[7:0])
 *   Cb signal must be treated in offset binary format 
 *   so that the binary interpretation retains monotonicity 
 *   from the minimum signal level to the maximum signal level.
 */
#define LWE22D_G2CMKEYL_0                                                   (0x18)
#define LWE22D_G2CMKEYL_0_CKRL                                              23:16
#define LWE22D_G2CMKEYL_0_CKGL                                              15:8
#define LWE22D_G2CMKEYL_0_CKBL                                              7:0


/*
 *   CKRU:
 *   R or Cr Color Chroma Key Upper Limit (CKRU[7:0])
 *   Cr signal must be treated in offset binary format 
 *   so that the binary interpretation retains monotonicity from 
 *   the minimum signal level to the maximum signal level.
 *  CKBU: This is B or Y color/chroma key upper limit value.
 *  
 *   CKGU:
 *   G or Cb Color/Chroma Key Upper Limit (CKGU[7:0])
 *   Cb signal must be treated in offset binary format 
 *   so that the binary interpretation retains monotonicity 
 *   the minimum signal level to the maximum signal level.
 */
#define LWE22D_G2CMKEYU_0                                                   (0x19)
#define LWE22D_G2CMKEYU_0_CKRU                                              23:16
#define LWE22D_G2CMKEYU_0_CKGU                                              15:8
#define LWE22D_G2CMKEYU_0_CKBU                                              7:0


/*
 *   StretchBLT, Source Image Data U base address A-buffer
 *   Start Address of  Source U-image Area, 4:2:0 Format.
 *   This parameter specifies the start address of source image stored in 
 *   the image buffer memory.  
 *   The [3:0] bits have to be 0, since memory client will assemble YUV into one
 *   422 format.
 */
#define LWE22D_G2UBA_A_0                                                    (0x1a)
#define LWE22D_G2UBA_A_0_SU1SA                                              31:0

/*
 *   Start Address of  Source V-image Area, 4:2:0 Format.
 *   This parameter specifies the start address of  source image stored in 
 *   the image buffer memory.  
 *   The [3:0] bits have to be 0, since memory client will assemble YUV into one
 *   422 format.
 */
#define LWE22D_G2VBA_A_0                                                    (0x1b)
#define LWE22D_G2VBA_A_0_SV1SA                                              31:0

/*
 *   RAISEFRAMEEN: ENABLE - SB needs to send a RAISE_FRAME control bit to EPP
 *   RAISEBUFER:   ENABLE - SB needs to send a RAISE_BUFFER control bit to EPP
 *   DIFMT:
 *  Destination Image Data FormatThis parameter defines the data format of distination output.  
 *   There are two groups of data formats, RGB and YCbCr (YUV) format.
 *   CbCr (UV) components may be represented in either offset binary or 2's complement.
 *  00000 U8Y8V8Y8, YUV 4:2:2, 8-bit for each component, U/V in offset binary format
 *  00001 Y8U8Y8V8, YUV 4:2:2, 8-bit for each component, U/V in offset binary format
 *  00010 Y8V8Y8U8, YUV 4:2:2, 8-bit for each component, U/V in offset binary format
 *  00011 V8Y8U8Y8, YUV 4:2:2, 8-bit for each component, U/V in offset binary format
 *  00100 U8Y8V8Y8, YUV 4:2:2, 8-bit for each component, U/V is 2's complement
 *  00101 Y8U8Y8V8, YUV 4:2:2, 8-bit for each component, U/V is 2's complement
 *  00110 Y8V8Y8U8, YUV 4:2:2, 8-bit for each component, U/V is 2's complement
 *  00111 V8Y8U8Y8, YUV 4:2:2, 8-bit for each component, U/V is 2's complement
 *  01000 bpp16 5-6-5 {R[4:0], G[5:0], B[4:0]} 
 *  01001 RESERVED
 *  01010 RESERVED
 *  01011 RESERVED
 *  01100 bpp16 5-6-5 Byte-swapped {G[2:0], B[4:0], R[4:0], G[5:3]}
 *  01101 RESERVED
 *  01110 R8G8B8A8
 *  01111 B8G8R8A8
 *  1xxxx RESERVED
 *   SIFMT:
 *   This parameter defines the data format of source input.  
 *   There are two groups of data formats, RGB format and YCbCr (YUV) format.
 *   CbCr (UV) components may be represented in either offset binary or 2's complement.
 *  00000 U8Y8V8Y8, YUV 4:2:2, 8-bit for each component, U/V in offset binary format
 *  00001 Y8U8Y8V8, YUV 4:2:2, 8-bit for each component, U/V in offset binary format
 *  00010 Y8V8Y8U8, YUV 4:2:2, 8-bit for each component, U/V in offset binary format
 *  00011 V8Y8U8Y8, YUV 4:2:2, 8-bit for each component, U/V in offset binary format
 *  00100 U8Y8V8Y8, YUV 4:2:2, 8-bit for each component, U/V is 2's complement
 *  00101 Y8U8Y8V8, YUV 4:2:2, 8-bit for each component, U/V is 2's complement
 *  00110 Y8V8Y8U8, YUV 4:2:2, 8-bit for each component, U/V is 2's complement
 *  00111 V8Y8U8Y8, YUV 4:2:2, 8-bit for each component, U/V is 2's complement
 *  01000 B5G6R5 5-6-5 {R[4:0], G[5:0], B[4:0]} 
 *  01001 RESERVED
 *  01010 RESERVED
 *  01011 RESERVED
 *  01100 B5G6R5 5-6-5 Byte-swapped {G[2:0], B[4:0], R[4:0], G[5:3]}
 *  01101 RESERVED
 *  01110 R8G8B8A8
 *  01111 B8G8R8A8
 *  1xxxx RESERVED
 *  
 *   StretchBlit Inputs 
 *   ==================
 *   RGB    inputs = {B5G6R5, B5G6R5BS, R8G8B8A8, B8G8R8A8}
 *   YUV420 input  = {YUV420 is colwerted into 4:2:2 UYVY via memory client}
 *   YUV422 inputs = {U8Y8V8Y8_OB, Y8U8Y8V8_OB, V8Y8U8Y8_OB, U8Y8V8Y8_OB, U8Y8V8Y8_TC, Y8U8Y8V8_TC, Y8V8Y8U8_TC, V8Y8U8Y8_TC}
 *  
 *   StretchBlit Outputs
 *   ===================
 *   RGB    outputs = {B5G6R5, B5G6R5BS, R8G8B8A8, B8G8R8A8}
 *   YUV422 outputs = {U8Y8V8Y8_OB, Y8U8Y8V8_OB, V8Y8U8Y8_OB, U8Y8V8Y8_OB, U8Y8V8Y8_TC, Y8U8Y8V8_TC, Y8V8Y8U8_TC, V8Y8U8Y8_TC}
 *  
 *   StretchBlit Input/Output Rules
 *   ==============================
 *  
 *  +--------------------------------------------------------------+--------------------+-----------------------------------------------------------+
 *  |    src format                                                | internal sb format |      dst format                                           |
 *  +--------------------------------------------------------------+--------------------+-----------------------------------------------------------+
 *  |  B5G6R5, B5G6R5BS, R8G8B8A8, B8G8R8A8                        |      R8G8B8A8      |  B5G6R5, B5G6R5BS, R8G8B8A8, B8G8R8A8                     |
 *  +--------------------------------------------------------------+--------------------+-----------------------------------------------------------+
 *  |  U8Y8V8Y8_*, Y8U8Y8V8_*, V8Y8U8Y8_*, U8Y8V8Y8_*, YUV420      |      YUV 4:4:4     |  U8Y8V8Y8_*, Y8U8Y8V8_*, V8Y8U8Y8_*, U8Y8V8Y8_*           |
 *  +--------------------------------------------------------------+--------------------+-----------------------------------------------------------+
 *  |  U8Y8V8Y8_*, Y8U8Y8V8_*, V8Y8U8Y8_*, U8Y8V8Y8_*, YUV420      |      YUV 4:4:4     |  B5G6R5, B5G6R5BS, R8G8B8A8, B8G8R8A8                     |
 *  +--------------------------------------------------------------+--------------------+-----------------------------------------------------------+
 *  
 *    StretchBlit color space colwerter supports YUV->RGB, YUV->YUV (gain) and RGB->RGB (gain).  There is no support for RGB->YUV.
 */
#define LWE22D_G2SBFORMAT_0                                                 (0x1c)
#define LWE22D_G2SBFORMAT_0_RAISEFRAMEVALUE                                 31:24
#define LWE22D_G2SBFORMAT_0_RAISEBUFFERVALUE                                23:16
#define LWE22D_G2SBFORMAT_0_RAISEFRAMEEN                                    15:15
#define LWE22D_G2SBFORMAT_0_RAISEBUFFEREN                                   14:14
#define LWE22D_G2SBFORMAT_0_RAISEBUFFEREN_DISABLE                           0x00000000
#define LWE22D_G2SBFORMAT_0_RAISEBUFFEREN_ENABLE                            0x00000001
#define LWE22D_G2SBFORMAT_0_DIFMT                                           12:8
#define LWE22D_G2SBFORMAT_0_DIFMT_U8Y8V8Y8_OB                               0x00000000
#define LWE22D_G2SBFORMAT_0_DIFMT_Y8U8Y8V8_OB                               0x00000001
#define LWE22D_G2SBFORMAT_0_DIFMT_Y8V8Y8U8_OB                               0x00000002
#define LWE22D_G2SBFORMAT_0_DIFMT_V8Y8U8Y8_OB                               0x00000003
#define LWE22D_G2SBFORMAT_0_DIFMT_U8Y8V8Y8_TC                               0x00000004
#define LWE22D_G2SBFORMAT_0_DIFMT_Y8U8Y8V8_TC                               0x00000005
#define LWE22D_G2SBFORMAT_0_DIFMT_Y8V8Y8U8_TC                               0x00000006
#define LWE22D_G2SBFORMAT_0_DIFMT_V8Y8U8Y8_TC                               0x00000007
#define LWE22D_G2SBFORMAT_0_DIFMT_B5G6R5                                    0x00000008
#define LWE22D_G2SBFORMAT_0_DIFMT_RESERVED9                                 0x00000009
#define LWE22D_G2SBFORMAT_0_DIFMT_RESERVED10                                0x0000000A
#define LWE22D_G2SBFORMAT_0_DIFMT_RESERVED11                                0x0000000B
#define LWE22D_G2SBFORMAT_0_DIFMT_B5G6R5BS                                  0x0000000C
#define LWE22D_G2SBFORMAT_0_DIFMT_RESERVED13                                0x0000000D
#define LWE22D_G2SBFORMAT_0_DIFMT_R8G8B8A8                                  0x0000000E
#define LWE22D_G2SBFORMAT_0_DIFMT_B8G8R8A8                                  0x0000000F
#define LWE22D_G2SBFORMAT_0_DIFMT_RESERVED16                                0x00000010
#define LWE22D_G2SBFORMAT_0_DIFMT_RESERVED17                                0x00000011
#define LWE22D_G2SBFORMAT_0_DIFMT_RESERVED18                                0x00000012
#define LWE22D_G2SBFORMAT_0_DIFMT_RESERVED19                                0x00000013
#define LWE22D_G2SBFORMAT_0_DIFMT_RESERVED20                                0x00000014
#define LWE22D_G2SBFORMAT_0_DIFMT_RESERVED21                                0x00000015
#define LWE22D_G2SBFORMAT_0_DIFMT_RESERVED22                                0x00000016
#define LWE22D_G2SBFORMAT_0_DIFMT_RESERVED23                                0x00000017
#define LWE22D_G2SBFORMAT_0_DIFMT_RESERVED24                                0x00000018
#define LWE22D_G2SBFORMAT_0_DIFMT_RESERVED25                                0x00000019
#define LWE22D_G2SBFORMAT_0_DIFMT_RESERVED26                                0x0000001A
#define LWE22D_G2SBFORMAT_0_DIFMT_RESERVED27                                0x0000001B
#define LWE22D_G2SBFORMAT_0_DIFMT_RESERVED28                                0x0000001C
#define LWE22D_G2SBFORMAT_0_DIFMT_RESERVED29                                0x0000001D
#define LWE22D_G2SBFORMAT_0_DIFMT_RESERVED30                                0x0000001E
#define LWE22D_G2SBFORMAT_0_DIFMT_RESERVED31                                0x0000001F
#define LWE22D_G2SBFORMAT_0_SIFMT                                           4:0
#define LWE22D_G2SBFORMAT_0_SIFMT_U8Y8V8Y8_OB                               0x00000000
#define LWE22D_G2SBFORMAT_0_SIFMT_Y8U8Y8V8_OB                               0x00000001
#define LWE22D_G2SBFORMAT_0_SIFMT_Y8V8Y8U8_OB                               0x00000002
#define LWE22D_G2SBFORMAT_0_SIFMT_V8Y8U8Y8_OB                               0x00000003
#define LWE22D_G2SBFORMAT_0_SIFMT_U8Y8V8Y8_TC                               0x00000004
#define LWE22D_G2SBFORMAT_0_SIFMT_Y8U8Y8V8_TC                               0x00000005
#define LWE22D_G2SBFORMAT_0_SIFMT_Y8V8Y8U8_TC                               0x00000006
#define LWE22D_G2SBFORMAT_0_SIFMT_V8Y8U8Y8_TC                               0x00000007
#define LWE22D_G2SBFORMAT_0_SIFMT_B5G6R5                                    0x00000008
#define LWE22D_G2SBFORMAT_0_SIFMT_RESERVED9                                 0x00000009
#define LWE22D_G2SBFORMAT_0_SIFMT_RESERVED10                                0x0000000A
#define LWE22D_G2SBFORMAT_0_SIFMT_RESERVED11                                0x0000000B
#define LWE22D_G2SBFORMAT_0_SIFMT_B5G6R5BS                                  0x0000000C
#define LWE22D_G2SBFORMAT_0_SIFMT_RESERVED13                                0x0000000D
#define LWE22D_G2SBFORMAT_0_SIFMT_R8G8B8A8                                  0x0000000E
#define LWE22D_G2SBFORMAT_0_SIFMT_B8G8R8A8                                  0x0000000F
#define LWE22D_G2SBFORMAT_0_SIFMT_RESERVED16                                0x00000010
#define LWE22D_G2SBFORMAT_0_SIFMT_RESERVED17                                0x00000011
#define LWE22D_G2SBFORMAT_0_SIFMT_RESERVED18                                0x00000012
#define LWE22D_G2SBFORMAT_0_SIFMT_RESERVED19                                0x00000013
#define LWE22D_G2SBFORMAT_0_SIFMT_RESERVED20                                0x00000014
#define LWE22D_G2SBFORMAT_0_SIFMT_RESERVED21                                0x00000015
#define LWE22D_G2SBFORMAT_0_SIFMT_RESERVED22                                0x00000016
#define LWE22D_G2SBFORMAT_0_SIFMT_RESERVED23                                0x00000017
#define LWE22D_G2SBFORMAT_0_SIFMT_RESERVED24                                0x00000018
#define LWE22D_G2SBFORMAT_0_SIFMT_RESERVED25                                0x00000019
#define LWE22D_G2SBFORMAT_0_SIFMT_RESERVED26                                0x0000001A
#define LWE22D_G2SBFORMAT_0_SIFMT_RESERVED27                                0x0000001B
#define LWE22D_G2SBFORMAT_0_SIFMT_RESERVED28                                0x0000001C
#define LWE22D_G2SBFORMAT_0_SIFMT_RESERVED29                                0x0000001D
#define LWE22D_G2SBFORMAT_0_SIFMT_RESERVED30                                0x0000001E
#define LWE22D_G2SBFORMAT_0_SIFMT_RESERVED31                                0x0000001F


/*
 *  output destination writes (dw) go either to image memory or epp
 *  0 Output data is sent to memory
 *  1 YUV or RGB data is passed directly to EPP module and no destination writes
 *    will take place.
 *   Enable Dithering (ENDITH)
 *   For 16 bit RGB output modes, the LSB of the color components can be
 *   modified by adding a variable residual value that will reduce the banding
 *   artifacts that can appear on the display.
 *   0 Normal operation
 *   1 Enable Dithering
 *   Key Signal Polarity (KPOL)
 *   Color/Chroma key signal is generated by comparing source input pixel 
 *   color to a range of a color specified by lower and upper limit values.
 *   The key signal is interpreted in two ways, depending on which one of 
 *   video and graphics images is foreground (and the other is background).  
 *   This is effective only if Key signal generator is enabled.
 *   (see G2CMKEYL, G2CMKEYU) 
 *   0 Key signal is set to 1 when source pixel is within the lower and upper 
 *   limit color range.
 *   1 Key signal is set to 1 when source pixel is outside the lower and upper 
 *   limit color range.
 *   Key Signal Generator Enable (KEYEN) Key signal generator generates either
 *   chroma key signal (from YCbCr signal) or color key signal (from RGB signal)
 *  0 Key signal generator is disabled.
 *  1 Key signal generator is enabled.
 *   StretchBLT Destination Buffer Selection (DBSEL)
 *   StretchBLT processing ilwolves frame-rate colwersion from a series of
 *   source images to another series of destination images (field-rate of 
 *   the source video to frame-rate of the PC display). In order to avoid 
 *   image tearing, it is preferred to use two buffer sections in the destination
 *   video area.  This bit selects one of the two buffering blocks to which this 
 *   StretchBLT command delivers the destination image. 
 *   The two buffering memory blocks are called A and B. 
 *  0 Destination image goes to ?A? buffer.
 *  1 Destination image goes to ?B? buffer.
 *   StretchBLT Source Buffer Selection (SBSEL)
 *   StretchBLT processing ilwolves frame-rate colwersion from a 
 *   series of source images to another series of destination images 
 *   (field-rate of the source video to frame-rate of the PC display).
 *   In order to avoid image tearing, it is preferred to use two buffer 
 *   sections in the source video area. This bit selects one of the two 
 *   buffering blocks from which this StretchBLT command receives the source image. 
 *   The two buffering memory blocks are called source-A and source-B. 
 *  0 Source image comes from ?source-A? buffer.
 *  1 Source image comes from ?source-B? buffer.
 *   StretchBLT Source Type (SITYPE)
 *   This bit identifies two types of source images. If source image is 2-to-1 
 *   interlaced and StretchBLT processes either one of the two interlaced 
 *   field-images, physical (positional) displacement between the two 
 *   interlaced fields must be taken into account.  One field-image that is 
 *   placed higher in position than the other field-image is called top-field 
 *   and the other is called bottom-field. StretchBLT processing has to 
 *   lower the top-field (or raise the bottom-field) to match the corresponding
 *   two target images in the overlay window (that is progressively scanned)
 *   right at the same position.   If source image is full frame-image 
 *   obtained from two interlaced field-images, its type is ?top-field?.  
 *   If source images are progressively scanned, the type identification is not 
 *   significant and they may be designated either one of the two types 
 *  0 Source image is ?top-field?.
 *  1 Source image is ?bottom-field?.
 *   RANGEREDFRM: 
 *  In the VC-1 specification, when the value of RANGEREDFRM variable (1-bit)
 *    for a picture is equal to 1, the picture shall be scaled up according 
 *   to the following equation:
 *  Y = clip( (( Y-128)*2) + 128);
 *  Cb = clip( ((Cb-128)*2) + 128);
 *  Cr = clip( ((Cr-128)*2) + 128);
 *  The clip operator limits the output to [0, 255].
 *  The input is also limited to [0, 255].
 *  In the VC-1 nomenclature, the output (Y, Cb, Cr) tuple corresponds to 
 *   the 'decoded' picture. The input (Y, Cb, Cr) tuple corresponds to 
 *   the 'reconstructed' picture. 
 *   The above equations create the decoded picture while keeping 
 *   the reconstructed picture intact.
 *  Only YUV _OB formats are supported with range reducation enabled,
 *   not the YUV _TC or RGB formats.
 *  ----------------------------------------------------
 *  At PortalPlayer this functionality was implemented in the DVO module. 
 *   The YUV data coming from the video frame buffers (reconstructed picture)
 *    in the SDRAM was scaled up using the above equations and given to the display. 
 *   the value of 1-bit wide RANGEREDFRM variable was configured in the DVO per frame 
 *   and the frame was scaled whenever RANGEREDFRM == 1. 
 *   HFTYPE:
 *   StretchBLT Horizontal Filter Mode (HFTYPE[2:0])
 *   The six-tap horizontal interpolation filter can be operated in 
 *   various operation modes. For the image expansion, it should be programmed 
 *   as a pure 6-tap interpolator.  For the image contraction, it can work as 
 *   partly lowpass filter and partly interpolater with varying degree depending 
 *   the contraction ratio.
 *   For StretchBLT, this parameter is used as an index (selection) 
 *   to an internal lookup table that stores the group of filter-coefficients 
 *  for the different modes of horizontal filtering. 
 *   000: Pure interpolation filter.
 *   001: 010 011 100 101 110: mix of interpolation and low pass filters
 *   111: DISABLE.
 *   VFEN:
 *   StretchBLT Vertical Filter Enable (VFEN)
 *   Vertical filter shall be disabled if source images come from host CPU 
 *   via CPU Read-FIFO. (SISEL)
 *   Vertical filter may be disabled to save some memory bandwidth but 
 *   this will likely result in degradation of image quality.  
 *   This option may also be used in the case where source image is progressive 
 *   scanning and there is no vertical scaling. 
 *  0 Vertical filter is disabled.
 *  1 Vertical filter is enabled.
 *   VFTYPE:
 *   StretchBLT Vertical Filter Mode (VFTYPE[1:0])
 *   This bit is effective when the Vertical Filter is enabled. (bit 18)
 *   The two-tap vertical interpolation filter can be operated in various modes. 
 *   For the image expansion, it should be programmed as a pure 2-tap 
 *   interpolator. For the image contraction, it can work as partly averager 
 *   and partly interpolator with varying degree depending on the contraction ratio.
 *   For StretchBLT, this parameter is used as an index (selection) to 
 *   an internal lookup table that stores the group of filter-coefficients for 
 *   the different modes of vertical filtering.
 *  00 Pure interpolation filter.
 *  01 25% averager, 75% interpolator.
 *  10 50% averager, 50% interpolator.
 *  11 100% averager.
 *   SBLTSRC:
 *   StretchBLT Source Selection  (SBLTSRC)
 *   CPU and VIP module can initiate StretchBLT operation. This bit indicates 
 *   which one initiates current StretchBLT command. Depending on the initiator, 
 *   appropriate source for the command exelwtion request is selected. Whichever 
 *   the initiator is, all the command parameters are supplied from the CPU (driver) 
 *   through the G2 command-FIFO and shadow registers.
 *  0 CPU is the command initiator.
 *  1 VIP module is the command initiator.
 *   SISEL:
 *   StretchBLT Source Provider Selection (SISEL)
 *   This bit indicates where the source images come from,  either the 
 *   image buffer memory or system memory of the host CPU.  If source 
 *  images come from host CPU, they come via CPU Read-FIFO. 
 *  0 Image buffer memory. 
 *  1 System memory of host CPU.
 *   SL:
 *   Source (input) Data Line Pairing  (SL) 
 *   This parameter specifies one of two cases of Y vs. U/V line-pairing 
 *   in the source data in 4:2:0 format.
 *   In the 4:2:0 format, there is one pair of U and V image-lines for 
 *   every two Y image-lines. 
 *  0 The first two Y image-lines share common pair of U and V image-lines. 
 *  1 The first two Y image-lines are paired with two different 
 *   pairs of U and V image-lines.
 *   UVST:
 *     00= 1/2 of Luma Buffer Stride; in this case
 *         Luma Buffer Stride should be multiple
 *         of 2 bytes.
 *     01= Equal to Luma Buffer Stride
 *     10= 1/4 of Luma Buffer Stride; in this case
 *         Luma Buffer Stride should be multiple
 *         of 4 bytes.
 *     11= use uvstride.
 *   Note: Use care when using a tiled surface, since 1/2 or 1/4
 *     of luma stride may yield an invalid pitch (stride).
 *   ENAHF:
 *   enable horizontal alpha filtering
 *   if disabled, use the alpha value of third tap for output pixel
 *  U,V line stride in 4:2:0  Format,
 *   IMODE:
 *   Source (input) Data Mode  (IMODE) This parameter defines the data mode of source input.  
 *   0: Multiplexed Mode (data format is specified by SIFMT[2:0])
 *   One block of source data in the Image Buffer memory contains 
 *   multiplexed component signals. 
 *   1: Planar mode, 4:2:0 YUV (SIFMT[2] specifies data format of U/V components)
 *   Three blocks of source data in the Image Buffer memory contain separate
 *   Y, U, and V component signals. 
 *  
 *    For cirlwlar buffer input to gr2d, the input format can not be planar.  It must be multiplex.
 *      i.e. no YUV420 planar cirlwlar buffer input to gr2d.
 */
#define LWE22D_G2CONTROLSB_0                                                (0x1d)
#define LWE22D_G2CONTROLSB_0_DISDW                                          31:31
#define LWE22D_G2CONTROLSB_0_ENDITH_DISABLE                                 0x00000000
#define LWE22D_G2CONTROLSB_0_ENDITH_ENABLE                                  0x00000001
#define LWE22D_G2CONTROLSB_0_ENDITH                                         30:30
#define LWE22D_G2CONTROLSB_0_KPOL                                           28:28
#define LWE22D_G2CONTROLSB_0_KPOL_WITHIN_BOUNDS                             0x00000000
#define LWE22D_G2CONTROLSB_0_KPOL_OUTSIDE_BOUNDS                            0x00000001
#define LWE22D_G2CONTROLSB_0_KEYEN                                          27:27
#define LWE22D_G2CONTROLSB_0_KEYEN_DISABLE                                  0x00000000
#define LWE22D_G2CONTROLSB_0_KEYEN_ENABLE                                   0x00000001
#define LWE22D_G2CONTROLSB_0_DBSEL                                          26:26
#define LWE22D_G2CONTROLSB_0_DBSEL_DST_A                                    0x00000000
#define LWE22D_G2CONTROLSB_0_DBSEL_DST_B                                    0x00000001
#define LWE22D_G2CONTROLSB_0_SBSEL                                          25:25
#define LWE22D_G2CONTROLSB_0_SBSEL_SRC_A                                    0x00000000
#define LWE22D_G2CONTROLSB_0_SBSEL_SRC_B                                    0x00000001
#define LWE22D_G2CONTROLSB_0_SITYPE                                         24:24
#define LWE22D_G2CONTROLSB_0_SITYPE_TOP_FIELD                               0x00000000
#define LWE22D_G2CONTROLSB_0_SITYPE_BOTTOM_FIELD                            0x00000001
#define LWE22D_G2CONTROLSB_0_RANGEREDFRM                                    23:23
#define LWE22D_G2CONTROLSB_0_RANGEREDFRM_DISABLE                            0x00000000
#define LWE22D_G2CONTROLSB_0_RANGEREDFRM_ENABLE                             0x00000001
#define LWE22D_G2CONTROLSB_0_HFTYPE                                         22:20
#define LWE22D_G2CONTROLSB_0_HFTYPE_LPF1                                    0x00000001
#define LWE22D_G2CONTROLSB_0_HFTYPE_LPF2                                    0x00000002
#define LWE22D_G2CONTROLSB_0_HFTYPE_LPF3                                    0x00000003
#define LWE22D_G2CONTROLSB_0_HFTYPE_LPF4                                    0x00000004
#define LWE22D_G2CONTROLSB_0_HFTYPE_LPF5                                    0x00000005
#define LWE22D_G2CONTROLSB_0_HFTYPE_LPF6                                    0x00000006
#define LWE22D_G2CONTROLSB_0_HFTYPE_DISABLE                                 0x00000007
#define LWE22D_G2CONTROLSB_0_DISCSC                                         19:19
#define LWE22D_G2CONTROLSB_0_DISCSC_ENABLE                                  0x00000000
#define LWE22D_G2CONTROLSB_0_DISCSC_DISABLE                                 0x00000001
#define LWE22D_G2CONTROLSB_0_VFEN                                           18:18
#define LWE22D_G2CONTROLSB_0_VFEN_DISABLE                                   0x00000000
#define LWE22D_G2CONTROLSB_0_VFEN_ENABLE                                    0x00000001
#define LWE22D_G2CONTROLSB_0_VFTYPE                                         17:16
#define LWE22D_G2CONTROLSB_0_VFTYPE_INTERP                                  0x00000000
#define LWE22D_G2CONTROLSB_0_VFTYPE_AVG25_INTERP75                          0x00000001
#define LWE22D_G2CONTROLSB_0_VFTYPE_AVG50_INTERP50                          0x00000002
#define LWE22D_G2CONTROLSB_0_VFTYPE_AVG                                     0x00000003
#define LWE22D_G2CONTROLSB_0_SBLTSRC                                        15:15
#define LWE22D_G2CONTROLSB_0_SBLTSRC_CPU                                    0x00000000
#define LWE22D_G2CONTROLSB_0_SBLTSRC_VIP                                    0x00000001
#define LWE22D_G2CONTROLSB_0_SISEL                                          13:13
#define LWE22D_G2CONTROLSB_0_SISEL_IMAGE_BUFFER                             0x00000000
#define LWE22D_G2CONTROLSB_0_SISEL_HOST                                     0x00000001
#define LWE22D_G2CONTROLSB_0_SL                                             10:10
#define LWE22D_G2CONTROLSB_0_SL_COMMON_PAIR                                 0x00000000
#define LWE22D_G2CONTROLSB_0_SL_DIFFERENT_PAIR                              0x00000001
#define LWE22D_G2CONTROLSB_0_UVST                                           9:8
#define LWE22D_G2CONTROLSB_0_UVST_UVS2X                                     0x00000000
#define LWE22D_G2CONTROLSB_0_UVST_UVS1X                                     0x00000001
#define LWE22D_G2CONTROLSB_0_UVST_UVS4X                                     0x00000002
#define LWE22D_G2CONTROLSB_0_UVST_UVS_G2UVSTRIDE                            0x00000003
#define LWE22D_G2CONTROLSB_0_ENAHF                                          7:7
#define LWE22D_G2CONTROLSB_0_ENAHF_DISABLE                                  0x00000000
#define LWE22D_G2CONTROLSB_0_ENAHF_ENABLE                                   0x00000001
#define LWE22D_G2CONTROLSB_0_ENAVF                                          6:6
#define LWE22D_G2CONTROLSB_0_ENAVF_DISABLE                                  0x00000000
#define LWE22D_G2CONTROLSB_0_ENAVF_ENABLE                                   0x00000001
#define LWE22D_G2CONTROLSB_0_IMODE                                          5:5
#define LWE22D_G2CONTROLSB_0_IMODE_MULTIPLEX                                0x00000000
#define LWE22D_G2CONTROLSB_0_IMODE_PLANAR                                   0x00000001
#define LWE22D_G2CONTROLSB_0_YUV422PLANAR                                   4:4
#define LWE22D_G2CONTROLSB_0_YUV422ROTATION                                 3:3


/*
 *    READWAIT
 *    Fast Rotate wait for read 0 = disable
 *   1 = enable
 *   Enabling this bit forces FR to wait for the reads to be stored
 *   in the data return fifo before started to send writes out to 
 *   the same block.
 *  
 *   Always set to ENABLE.  This is debug only.
 *    FR_TYPE:
 *    Fast Rotate type     
 *   000 = FLIP_X 
 *   001 = FLIP_Y 
 *   010 = TRANS_LR (mirrors about diagonal. Diagonal runs from upper left to lower right)
 *   011 = TRANS_RL (mirrors about diagonal. Diagonal runs from upper right to lower left)
 *   100 = ROT_90 (counter clock wise by 90 degrees)
 *   101 = ROT_180 
 *   110 = ROT_270 (clock wise by 90 degrees)
 *   111 = IDENTITY
 *   FR_MODE:
 *   Fast Rotate mode sel
 *  ------------------------------------------------------------------
 *   G2 Fast Rotate
 *  
 *   Transforms a surface via FR_TYPE transformation.
 *   Works in either 2-buffer (copy)  or 1-buffer mode (in place) mode.
 *   The engine breaks down a larger surface into a grid of smaller FR_BLOCKs.
 *   Works on the granularity of an FR_BLOCK.  An FR_BLOCK is:
 *     16x16 pixel block (DSTCD = bpp8)
 *      8x8  pixel block (DSTCD = bpp16)
 *      4x4  pixel block (DSTCD = bpp32)
 *   Max surface size is 4096x4096 
 *  
 *   Key information
 *    - source and destination base address must be 128-bit word aligned
 *    - engine works on FR_BLOCK granularity:
 *        transformed surface width  in multiples of 16-bytes**
 *        transformed surface height in multiples of 16/8/4 lines for bpp8/bpp16/bpp32 FR_BLOCK
 *        if surface dimension is not a multiple, sw can program FR engine to transform larger surface 
 *           (round up to next FR_BLOCK in width and height-->transform-->callwlate relative memory pointer address)
 *    - during a rotational transformation (TRANS_LR, TRANS_RL, ROT_90, ROT_270):
 *        the stride of the output surface != the stride of the input surface when working on non-square input
 *        input:                        output:
 *        *^----------------------           *$----------------------
 *        $ 0  1  2  3  4  5  6  7           ^ 24 16 8  0  ^  ^  ^  ^
 *        | 8  9 10 11 12 13 14 15           | 25 17 9  1  ^  ^  ^  ^
 *        |16 17 18 19 20 21 22 23           | 26 18 10 2  ^  ^  ^  ^
 *        |24 25 26 27 28 29 30 31           | 27 19 11 3  ^  ^  ^  ^ 
 *        | -  -  -  -  *  *  *  *           | 28 20 12 4  *  *  *  *
 *        | -  -  -  -  *  *  *  *           | 29 21 13 5  *  *  *  *
 *        | -  -  -  -  *  *  *  *           | 30 22 14 6  *  *  *  *
 *        | -  -  -  -  *  *  *  *           | 31 23 15 7  *  *  *  *
 *  
 *    - Tiling alignment restrictions subsume FR restrictions because the 
 *      FR surface base address is the same as the start address.
 *  
 *    Register Programming
 *    ====================
 *     FR_MODE - inplace or copy
 *     FR_TYPE - type of transformation
 *     DSTCD - bpp8, bpp16, bpp32
 *     SRCBA - source base address
 *     SRCWIDTH - (width in pixels-1)
 *     SRCHEIGHT - (height in lines-1)
 *     SRCS - source stride 
 *     DSTBA - dest base address
 *     DSTS - dest stride
 *     FR_READWAIT - always set to enable
 *  
 *   FR inefficency in the following setup :
 *   1. FR_MODE==SQUARE
 *   2. if(FR_TYPE==YFLIP && SRCHEIGHT==m*n && n==odd number) where m=8bpp?16:16bpp?8:4
 *      Or if(FR_TYPE==XFLIP && SRCWIDTH==m*n && n==odd number) where m=8bpp?16:16bpp?8:4
 *   3. Then, the middle 8 lines/pixels, E.G. YFLIP/16bpp/n==3, line8 to line15 will be processed twice. 00 = disable fast rotate - this turns off the 2nd level clock to fr_rotate engine.  Please remember to do so!
 *   01 = src/dst copy mode - two separate buffers
 *   10 = square in place - one buffer
 *   11 = blank
 *   CLIPC:
 *   Clipping rectangle control, 
 *   if clip enable, bit 57 XYTDW should be cleared. 
 *   0x=clipping disabled, 
 *   10=draw only inside clipping rectangle, 
 *   11=draw only outside clipping rectangle
 *  VCAA safe mode, if turned on, all up/bottom color data will be read in. This is a debug/cya incase the top/bottom color
 *    fetch is broken.
 *   SWAPBLT
 *   Used by Palm OS to highlight a selected icon by swapping
 *    Background and Foreground colors in a rectangle area
 *   PXLREP: Pixel replication for Palm OS.
 *   ALPSRCORDST: 32bits blending mode, output alpha selection 0:source alpha, 1:destination alpha
 *   Alpha blending method
 *   FIX:     
 *     ALPHA blending:  Fixed alpha,  ALPHA is the value, B5G6R5 * B5G6R5
 *     VCAA:            B5G6R5->B5G6R5
 *   PL1BPP:  
 *     ALPHA blending:  Alpha 1bit/pixel from memory plane, B5G6R5 * B5G6R5
 *        NOTE: when ALPTYPE is PL1BPP, DSTX[2:0] must be zero, see bug 344489
 *   PL2BPP:    
 *     ALPHA blending:  Alpha 2bits/pixel from memory plane, B5G6R5 * B5G6R5
 *        NOTE: when ALPTYPE is PL2BPP, DSTX[1:0] must be zero, see bug 344489
 *   PL4BPP:    
 *     ALPHA blending:  Alpha 4bits/pixel from memory plane, B5G6R5 * B5G6R5
 *        NOTE: when ALPTYPE is PL4BPP, DSTX[0:0] must be zero, see bug 344489
 *   PL8BPP:    
 *     ALPHA blending:  Alpha 8bits/pixel from memory plane, B5G6R5 * B5G6R5
 *   PL44BPP:   
 *     ALPHA blending:  Alpha 8bits/pixel from memory plane src*4bits+dst*4bits, B5G6R5 * B5G6R5
 *   PLS1BPP:   
 *     ALPHA blending:  Alpha 1bit from source B5G5R5A1, alpha(MSB). Dest: B5G6R5. 
 *     VCAA:            reserved
 *   PLS4BPPAL: 
 *     ALPHA blending:  Alpha 4bits from source A4B4G4R4, alpha(LSB). Dest: B5G6R5.
 *   PLS4BPP:   
 *     ALPHA blending:  Alpha 4bits from source B4G4R4A4, alpha(MSB). Dest: B5G6R5.
 *     VCAA:            R8G8B8A8->B5G6R5 without reading VCAA plane
 *                       (surface blit with bpp down colwert - implemented in hw by vcaa engine; not really a vcaa resolve)
 *   PLS8BPP:   
 *     ALPHA blending:  Alpha 8bits from source/destination(decided by ALPSRCORDST), 
 *                      R8G8B8A8, alpha(MSB). DST: R8G8B8A8
 *     VCAA:            R8G8B8A8->R8G8B8A8, alpha has same blending method as RGB
 *   PLS8BX:    
 *     ALPHA blending:  Alpha 8bits from source B8G8R8A8, alpha(MSB). Dest: B5G6R5. (**Restrictions)
 *     VCAA:            R8G8B8A8->B5G6R5
 *   PLS1BPPAL: 
 *     ALPHA blending:  Alpha 1 bit from source A1B5G5R5, alpha(LSB). Dest: B5G6R5
 *     VCAA:            A1B5G5R5->A1B5G5R5
 *   **Restriction
 *   PLS8BX alpha blending has the following restrictions
 *   1. Source/destination addresses have to be in 128bit boundary.
 *   2. Destination width has to be multiple of 4 pixels.
 *   3. Source/Destination strides have to be multiple of 128bits.
 *   BEWSWAP
 *   Host port word swap 1=enable   0= disable
 *   BEBSWAP
 *   Host port byte swap 1=enable   0= disable
 *     1= enable
 *   BITSWAP
 *   Host port bit swap 1=enable   0= disable
 *     1= enable
 */
#define LWE22D_G2CONTROLSECOND_0                                            (0x1e)
#define LWE22D_G2CONTROLSECOND_0_FR_READWAIT_DISABLE                        0x00000000
#define LWE22D_G2CONTROLSECOND_0_FR_READWAIT_ENABLE                         0x00000001
#define LWE22D_G2CONTROLSECOND_0_FR_READWAIT                                29:29
#define LWE22D_G2CONTROLSECOND_0_FR_TYPE                                    28:26
#define LWE22D_G2CONTROLSECOND_0_FR_TYPE_FLIP_X                             0x00000000
#define LWE22D_G2CONTROLSECOND_0_FR_TYPE_FLIP_Y                             0x00000001
#define LWE22D_G2CONTROLSECOND_0_FR_TYPE_TRANS_LR                           0x00000002
#define LWE22D_G2CONTROLSECOND_0_FR_TYPE_TRANS_RL                           0x00000003
#define LWE22D_G2CONTROLSECOND_0_FR_TYPE_ROT_90                             0x00000004
#define LWE22D_G2CONTROLSECOND_0_FR_TYPE_ROT_180                            0x00000005
#define LWE22D_G2CONTROLSECOND_0_FR_TYPE_ROT_270                            0x00000006
#define LWE22D_G2CONTROLSECOND_0_FR_TYPE_IDENTITY                           0x00000007
#define LWE22D_G2CONTROLSECOND_0_FR_MODE                                    25:24
#define LWE22D_G2CONTROLSECOND_0_FR_MODE_DISABLE                            0x00000000
#define LWE22D_G2CONTROLSECOND_0_FR_MODE_SRC_DST_COPY                       0x00000001
#define LWE22D_G2CONTROLSECOND_0_FR_MODE_SQUARE                             0x00000002
#define LWE22D_G2CONTROLSECOND_0_FR_MODE_BLANK                              0x00000003
#define LWE22D_G2CONTROLSECOND_0_CLIPC                                      22:21
#define LWE22D_G2CONTROLSECOND_0_G2NOSTOP                                   20:20
#define LWE22D_G2CONTROLSECOND_0_VCAA_SAFE_MODE                             16:16
#define LWE22D_G2CONTROLSECOND_0_SWAPBLT                                    15:15
#define LWE22D_G2CONTROLSECOND_0_PXLREP                                     14:14
#define LWE22D_G2CONTROLSECOND_0_PXLREP_DISABLE                             0x00000000
#define LWE22D_G2CONTROLSECOND_0_PXLREP_ENABLE                              0x00000001
#define LWE22D_G2CONTROLSECOND_0_ALPSRCORDST                                9:9
#define LWE22D_G2CONTROLSECOND_0_ALPSRCORDST_DISABLE                        0x00000000
#define LWE22D_G2CONTROLSECOND_0_ALPSRCORDST_ENABLE                         0x00000001
#define LWE22D_G2CONTROLSECOND_0_ALPTYPE                                    8:4
#define LWE22D_G2CONTROLSECOND_0_ALPTYPE_FIX                                0x00000000
#define LWE22D_G2CONTROLSECOND_0_ALPTYPE_PL1BPP                             0x00000001
#define LWE22D_G2CONTROLSECOND_0_ALPTYPE_PL2BPP                             0x00000002
#define LWE22D_G2CONTROLSECOND_0_ALPTYPE_PL4BPP                             0x00000003
#define LWE22D_G2CONTROLSECOND_0_ALPTYPE_PL8BPP                             0x00000004
#define LWE22D_G2CONTROLSECOND_0_ALPTYPE_PL44BPP                            0x00000005
#define LWE22D_G2CONTROLSECOND_0_ALPTYPE_PLS1BPP                            0x00000006
#define LWE22D_G2CONTROLSECOND_0_ALPTYPE_PLS4BPPAL                          0x00000007
#define LWE22D_G2CONTROLSECOND_0_ALPTYPE_PLS4BPP                            0x00000008
#define LWE22D_G2CONTROLSECOND_0_ALPTYPE_PLS8BPP                            0x00000009
#define LWE22D_G2CONTROLSECOND_0_ALPTYPE_PLS8BX                             0x0000000A
#define LWE22D_G2CONTROLSECOND_0_ALPTYPE_PLS1BPPAL                          0x0000000B
#define LWE22D_G2CONTROLSECOND_0_BEWSWAP                                    3:3
#define LWE22D_G2CONTROLSECOND_0_BEWSWAP_DISABLE                            0x00000000
#define LWE22D_G2CONTROLSECOND_0_BEWSWAP_ENABLE                             0x00000001
#define LWE22D_G2CONTROLSECOND_0_BEBSWAP                                    2:2
#define LWE22D_G2CONTROLSECOND_0_BEBSWAP_DISABLE                            0x00000000
#define LWE22D_G2CONTROLSECOND_0_BEBSWAP_ENABLE                             0x00000001
#define LWE22D_G2CONTROLSECOND_0_BITSWAP                                    1:1
#define LWE22D_G2CONTROLSECOND_0_BITSWAP_DISABLE                            0x00000000
#define LWE22D_G2CONTROLSECOND_0_BITSWAP_ENABLE                             0x00000001



/*
 *  ----------------------------------------------------------------------
 *  
 *   G2 Blit Formats:
 *   (1) G2 Blit size is based on height in lines, width in pixels,
 *       stride in bytes, and pixel size (color depth). Unless alpha
 *       blending is enabled, G2 doesn't care about pixel components.
 *  
 *       Source color depths supported by G2 Blit:
 *       =========================================
 *         color depth same as destination: G2CONTROLMAIN.SRCCD = 1
 *         monochrome:                      G2CONTROLMAIN.SRCCD = 0
 *         (source monochrome color determined by G2SRCBGC, G2SRCFGC)
 *  
 *       Pattern color depths supported by G2 Blit:
 *       =========================================
 *         color depth same as destination: G2PATOS.PATCD = 1
 *         monochrome:                      G2PATOS.PATCD = 0
 *         (pattern monochrome color determined by G2PATBGC, G2PATFGC)
 *  
 *       Destination color depths supported by G2 Blit:
 *       ==============================================
 *         1 byte per pixel:    G2CONTROLMAIN.DSTCD = BPP8
 *         2 bytes per pixel:   G2CONTROLMAIN.DSTCD = BPP16
 *         4 bytes per pixel:   G2CONTROLMAIN.DSTCD = BPP32
 *  
 *   (2) Alpha blending operates on 3 or 4 component pixels of a specified
 *       color depth. One of the components may be alpha (A), depending on the
 *       alpha blend format, which is only allowed in certain component
 *       positions defined below. The only format rule of the remaining 
 *       components is that they must be in the same source and destination
 *       position because the alpha blend engine does not support component
 *       swapping. For example, valid 32BPP formats are: RGBA_8888->RGBA_8888,
 *       BGRA_8888->BGRA_8888, GRBA_8888->GRBA_8888, etc.
 *  
 *       Source formats supported by G2 alpha blend:
 *       ===========================================
 *         color depth BPP8:  no supported formats
 *         color depth BPP16: 
 *            ALPTYPE_PLS1BPP: xxxA_5551 (ex: BGRA_5551 = A[15] R[14:10] G[9:5] B[4:0])
 *            ALPTYPE_PLS4BPP: xxxA_4444 (ex: BGRA_4444 = A[15:12] R[11:8] G[7:4] B[3:0])
 *            (other)        : xxx_565   (ex: BGR_565   = R[15:11] G[10:5] B[4:0])
 *         color depth BPP32: 
 *            (all)          : xxxA_8888 (ex: BRGA_8888 = A[31:24] R[23:16] G[15:8] B[7:0])
 *  
 *  
 *       Destination formats supported by G2 alpha blend:
 *       ================================================
 *         color depth BPP8:  no supported formats
 *         color depth BPP16: xxx_565   (ex: BGR_565   = R[15:11] G[10:5] B[4:0])
 *         color depth BPP32: xxxA_8888 (ex: BRGA_8888 = A[31:24] R[23:16] G[15:8] B[7:0])
 *  
 *       The following table 
 *       +---------+-------------------------------+---------+----------------+--------------------------------------------+
 *       | src cd  |   src format                  | dst cd  | dst format     | examples:                                  |
 *       +---------+-------------------------------+---------+----------------+--------------------------------------------+
 *       |  BPP8   |  no support                   |  BPP8   | no support     |                                            |
 *       +---------+-------------------------------+---------+----------------+--------------------------------------------+
 *       |  BPP16  | xyzA_5551, xyzA_4444, xyz_565 |  BPP16  | xyz_565        | BGRA_5551->BGR_565, RGB_565->RGB565        |
 *       +---------+-------------------------------+---------+----------------+--------------------------------------------+
 *       |  BPP32  | xyzA_8888                     |  BPP32  | xyzA_8888      | RGBA_8888->RGBA_8888, BGRA_8888->BGRA_8888 |
 *       +---------+-------------------------------+---------+----------------+--------------------------------------------+
 *       |  BPP32  | xyzA_8888                     |  BPP16  | xyz_565        | RGBA_8888->RGB_565, BGRA_8888->BGR_565     |
 *       +---------+-------------------------------+---------+----------------+--------------------------------------------+
 *  
 *  ----------------------------------------------------------
 *   VCAA Resolve engine
 *  
 *   (1) Format support 
 *  
 *   +-----------------+--------------------+----------------------+
 *   | srcColor format | srcCoverage format | legal dest format(s) |
 *   +-----------------+--------------------+----------------------+
 *   |     A1B5G5R5    |        VCAA        |       A1B5G5R5       |
 *   +-----------------+--------------------+----------------------+
 *   |     R8G8B8A8    |        VCAA        |   R8G8B8A8, B5G6R5   |
 *   +-----------------+--------------------+----------------------+
 *   |      B5G6R5     |        VCAA        |         B5G6R5       |
 *   +-----------------+--------------------+----------------------+
 *  
 *   To program resolve type:
 *     ALPTYPE == FIX:       B5G6R5   -> B5G6R5
 *     ALPTYPE == PLS1BPPAL: A1B5G5R5 -> A1B5G5R5
 *     ALPTYPE == PLS8BPP:   R8G8B8A8 -> R8G8B8A8
 *     ALPTYPE == PLS8BX:    R8G8B8A8 -> B5G6R5
 *  
 *   (2) Color surface programming
 *  
 *       color surface base address:  SRCBA
 *         sub-surface start x:       SRCX      (pixel index)
 *         sub-surface start y:       SRCY      (line index)
 *       color surface stride:        SRCS      (bytes)
 *  
 *       The engine supports sub-surface resolve.  One can imagine a full color buffer of 1024x768, but
 *   
 *       [pseudo_code]
 *         color_surface_start_address_fetch = plwrctx->regs.rG2SRCBA.uSRCBA() +
 *          (plwrctx->regs.rG2SRCPS.uSRCX() * VCAAState.color_surface_depth) + 
 *          (plwrctx->regs.rG2SRCPS.uSRCY() * VCAAState.color_surface_stride);
 *  
 *   (3) VCAA surface programming
 *  
 *       vcaa surface base address:   PATBA
 *       vcaa surface stride:         PATST     (bytes)
 *  
 *       [pseudo_code]
 *         vcaa_surface_start_address_fetch = plwrctx->regs.rG2PATBA.uPATBA();
 *  
 *         SW *MUST* directly program this register to the proper sub-surface location in the vcaa surface
 *          which corresponds to the SRCX and SRCY programming. The callwlation is
 *              PATBA = vcaa_base_address + SRCX * (1) + SRCY * PATST; 
 *  
 *   (4) Resolve surface programming
 *  
 *       resolve window width:        DSTWIDTH  (pixels)
 *       resolve window height:       DSTHEIGHT (lines)
 *       resolve output stride:       DSTS      (bytes)
 *  
 *    These values program the width and height of the resolved surface.  In the base case, 
 *     DSTWIDTH  = color surface width 
 *     DSTHEIGHT = color sufrace height
 *     DSTS      = resolve surface format bpp * DSTWIDTH
 *  
 *    The vcaa engine technically supports DSTX and DSTY (like its cousin SRCX and SRCY).  This was not
 *    requested by the original RFE by SW during the initial design.  
 *   
 *    [pseudo_code]
 *       resolve_surface_start_address_put = plwrctx->regs.rG2DSTBA.uDSTBA() +
 *         (plwrctx->regs.rG2DSTPS.uDSTX() * resolve_surface_color_depth) +
 *         (plwrctx->regs.rG2DSTPS.uDSTY() * resolve_surface_color_stride);
 *  
 *  
 *   (5) Resolving a pixel
 *  
 *     The coverage surface is a 8 bits per pixel surface (C4X4).  The surface is initialized to all 1s by 3d.
 *  
 *     For a given pixel, we need to look up 4 bits of coverage data.  
 *     If the data is 0xf, the pixel does not need to be resolved.
 *     Otherwise, we need to reblend the pixel with certain weightings of its neighbors (i.e. we're on the edge of some geometry)
 *     
 *     If any coverage bits is 0, we need to resolve the color.  To do so, we callwlate the following eqn:
 *     color_new = (
 *                   20*color_old +
 *                   27*(
 *                       ((cover_down  == 0) ? color_down  : color_old) +
 *                       ((cover_right == 0) ? color_right : color_old) +
 *                       ((cover_left  == 0) ? color_left  : color_old) +
 *                       ((cover_up    == 0) ? color_up    : color_old) 
 *                       )
 *                 ) / 128;
 *  
 *     If coverage bit == 0, then we use the neighbor color value in the resolve
 *     If coverage bit == 1, then we use the center   color value in the resolve
 *  
 *     The resolve is done on a per channel (r/g/b/a) basis:
 *       format colwert input channel to 8-bit
 *       perform resolve equation above
 *       format colwert to output format
 *  
 *     For the resolution of the alpha channel:
 *       In B5G6R5:   no alpha information.  do nothing.
 *       In A1B5G5R5: no resolve.  keep the source color alpha bit.
 *       In R8G8B8A8: normal 8-bit channel resolve
 *  
 *   (6) Resolving a pixel on the edge
 *  
 *      When on an edge, the vcaa engine treats the coverage bit as a 1.  Use the center color value.
 *      This holds true even on the edge of the sub-surface resolve window when located in the middle of the larger surface
 *  
 *   (7) Maximum dimensions
 *    
 *      The maximum resolve sub-surface is 4096 pixels wide.
 *  
 *   (8) Surface restrictions
 *      The VCAA engine has the following restrictions w.r.t input surfaces
 *        - input base address is 128-bit memory word aligned   
 *        - input stride is multiple of 128-bits (16 bytes)
 *      The VCAA engine has the following restrictions w.r.t output surfaces
 *        - output base address is 128-bit memory word aligned
 *        - output stride is multiple of 128-bits (16 bytes)
 *   (9) Coverage surface layout
 *      The surface is C4X4.  doc/<previous projects>/3d/qrast_sides.vsd/qrast_arch.doc/qrast_vcaa.doc has a wealth of info.
 *  
 *      Surface layout (C4X4)
 *      =====================
 *  
 *         (lsb2msb)  C0_C1_C2_C3_X_X_X_X     (X is a don't care.  Qrast inits to 1)
 *         (msb2lsb)  X_X_X_X_C3_C2_C1_C0
 *  
 *  
 *       +          o          o
 *            C0
 *               C1   Cov(0,1)
 *          C2
 *            C3
 *       o          o          o
 *       
 *         Cov(1,0)   Cov(1,1)
 *  
 *       o          o          o
 *  
 *  
 *       +/o = pixel centers
 *  
 *       In the example above,  
 *          Cov(0,0) = X_X_X_X_C3_C2_C1_C0
 *          C3       = V_C3(Cov(0,0));
 *          ..         ..
 *          C0       = V_C0(Cov(0,0));
 *       #define V_C3(bits) (bits & 0x8) 
 *       #define V_C2(bits) (bits & 0x4) 
 *       #define V_C1(bits) (bits & 0x2) 
 *       #define V_C0(bits) (bits & 0x1) 
 *   
 *      Resolving a pixel
 *      =================
 *       In an ideal world, the coverage would be simply laid out such that when resolving Pixel(x,y), we would
 *      need only to fetch Cov(x,y) to get all 4 bits of information.
 *       In actuality, the coverage surface was rearranged to help qrast in performance.  To resolve Pixel(x,y)
 *      gr2d has to fetch 4 C4X4 bytes [Cov(x,y), Cov(x-1,y), Cov(x,y-1), Cov(x-1,y-1)].  From each C4X4 byte, it
 *      extracts one bit which is used in the resolve equations shown in (5).
 *  
 *  
 *      Need the logical vcaa bits (DRLU) for blend which is spread across four C4X4 coverage bytes
 *  
 *            U
 *      L   
 *          o
 *              R
 *        D
 *  
 *      where
 *  
 *       o = Pixel(x,y)
 *       D = V_C1(Cov(x-1,y))
 *       R = V_C0(Cov(x,y))
 *       L = V_C3(Cov(x-1,y-1))
 *       U = V_C2(Cov(x,y-1))
 *  
 *  
 *   (10) Other documentation
 *         cmod/g2/fsim/g2vcaa.cpp - cmodel implementation with comments
 *         doc/<previous projects>/3d/qrast_sides.vsd/qrast_arch.doc/qrast_vcaa.doc  
 *         
 *
 *   PATSEL
 *   pattern Data Select: SRCSEL and PATSEL can't be both enabled.
 *   DST_RD_WR_SEP:
 *   This feature is not offically supported.
 *  seperate destination read/write surface address
 *  0: read/write use DSTBA
 *  1: read uses SRCBA_B, write uses DSTBA
 *   DSTDIR
 *  destination direct addressing
 *   SRCDIR:
 *  source direct addressing
 *  GCSW:
 *  Display Switching Window Control (GCSW[1:0])
 *  This parameter controls multi-buffering for Display.
 *  x0 At end of current command, don't send signal to Display to switch buffer.
 *  01 two buffers, dstba and dstba_b are used
 *  11 three buffers, dstba, dstba_b dstba_c are used
 *   SRCBAS
 *   Source base address select: 
 *    0=srcba, 1=dstba. This is not used for Line Draw and 
 *   if source data comes from host memory.
 *   SRCT:
 *   Source transparency enable: 0x=source transparency disabled, 
 *   10=mono source background transparency 
 *      or color source transparency, 
 *   11=mono source foreground transparency 
 *      or ilwerse color source transparency. 
 *   NOTE: When source transparency is enabled and SRCCD==0(mono)
 *         SRCBGC!=SRCFGC should be satisfied.
 *   HLMONO:
 *   Start from Msb or lsb in byte when mono expansion
 *   If HLMONO is 1, bit 0 (the lsb) is the first bit
 *   If HLMONO is 0, bit 7 (the msb) is the first bit
 *   SRCCD:
 *  0 Source mono
 *  1 Source has same color depth as destination
 *    SRCCD==0 xdir/ydir has to be 0
 *   DSTT:
 *   Destination read transparency enable:
 *     0x=destination read transparency disabled
 *     10=color destination read transparency
 *     11=ilwerse color destination read transparency.
 *   20 rw INIC                   
 *   Initiate Command
 *   (1=initiate command immediately, 0=wait for launch write)
 *   YFLIP:
 *   flip y direction to make image upside down or the other way.
 *   If YFLIP==1, ROP can not include destination.
 *   PATPACK:
 *   Pattern Data is in Pack Mode. 
 *    PATLNGAP in G2PATPACK is the line gap for pattern packed mode 
 *    If(PATPACK && ~PATSEl), pattern data is packed and from screen, PATMONOW/PATMONOH 
 *    should be programmed properly to fetch pattern data from 
 *    frame buffer. 
 *    Note, PACK is not offically supported when the surface is tiled.
 *   SRCPACK:
 *   Source Data is in Pack Mode. 
 *    SRCLNGAP in G2SRCPACK is the line gap for source packed mode.
 *    If(SRCPACK && ~SRCSEL), source data is packed and from screen, SRCMONOW/SRCMONOH 
 *    should be programmed properly to fetch data from 
 *    frame buffer.
 *    Note, PACK is not offically supported when the surface is tiled.
 *   XYTDW:
 *   xy transpose,
 *   Line stride DSTS has to be 16bytes aligned if enabled.  
 *   If XYTDW==1, ROP can not include destination.
 *    YDIR:
 *    0=incrementing, 1=decrementing.
 *    ydir should not be set when source surface has different color depth as destination surface.
 *    SRCCD==0 (mono src) OR PATCD==0 (mono pat), ydir has to be 0
 *    XDIR:
 *    0=incrementing, 1=decrementing.
 *    xdir should not be set when source surface has different color depth as destination surface.
 *    e.g. ALPTYPE=PLS8BX (32bpp blending with 16bpp), xdir has to be 0
 *    SRCCD==0 (mono src) OR PATCD==0 (mono pat), xdir has to be 0
 *   PATFL:
 *   When mono pattern is set, we use mono tile pattern fill.
 *  
 *    current pattern client can support a 16x16 bit tile which can be used
 *    to generate a larger surface (i.e. tile this surface in the x and y direction)
 *    The tile is stored in memory.  Lwrrently, PATXO and PATYO tell you where
 *    to start in the 16x16 tile when expanding the monochrome data.  
 *  
 *    See patxo and patyo comments below for programming (hw bid 247332)
 *     When xdir==1
 *       Patxo = pattern width - (patxo+destination width)&0xF   -- For xoffsets
 *    y offsets PATYO remains the same when xdir/ydir change.
 *  
 *  
 *    How the tile replication pattern looks based on xdir/ydir (without programming patxo above):
 *  
 *    xdir=0 ydir=0              xdir=1 ydir=0
 *     
 *     +---+--+                  +--+---+
 *     |^|^|^||                  ||^|^|^|
 *     |~~~|~~|                  |~~|~~~|
 *     +---+--|                  |--+---+
 *     |^|^|^||                  ||^|^|^|
 *     +------+                  +------+
 *  
 *                                     
 *     +---+--+                  +--+---+
 *     |^|^|^||                  ||^|^|^|
 *     |~~~|~~|                  |~~|~~~|
 *     +---+--|                  |--+---+
 *     |^|^|^||                  ||^|^|^|
 *     +------+                  +------+
 *      //
 *    xdir=0 ydir=1              xdir=1 ydir=1
 *  
 *     Mono tile is 
 *        +---+
 *        |^|^|
 *        |~~~|
 *        +---+
 *  
 *  
 *   PATSLD
 *   BitBlt Solid Pattern Fill: 1=enable. BGC will be used as 
 *   the color value.
 *   SRCSLD:
 *   BitBlt Solid Source Color Fill: 1=enable. FGC will be used as 
 *   the color value.
 *   ALPEN:
 *   BitBlt Alpha Blending, 1=enable. 0=disable,
 *  when both Faden and alpen are 1, output=Source*alpha_v + fadoff, 
 *   alpha_v is decided by alptype
 *   FADEN
 *   BitBlt Source Copy Fade enable, 1=enable (share with mltln), only support
 *    16bpp mode
 *  TESTOBIT:
 *  Command finish timing bit 0: 2D command finishs when last data has been pushed to memory write client.
 *   1: 2D command waits memory write client to be idle to finish.
 *  TURBOFILL:
 *  fast fill rectangle in 128bit/clock
 *  Some limitaions with this mode:
 *  srcsld==1 rop==0xcc, no clipping, no transparency
 *  xdir==0, ydir==0, flip==0, xytdw==0
 *  Results are undefined if the above limitations are not satified.
 *   CMDT
 *   Command Type:  00=BitBlt
 *   01=Line Draw 
 *   10=VCAA
 *   11=reserved
 *    When the raise command is in exelwting  
 *    (there are no other outstanding commands with same channel being exelwted)
 */
#define LWE22D_G2CONTROLMAIN_0                                              (0x1f)
#define LWE22D_G2CONTROLMAIN_0_PATSEL                                       30:30
#define LWE22D_G2CONTROLMAIN_0_PATSEL_SCREEN                                0x00000000
#define LWE22D_G2CONTROLMAIN_0_PATSEL_MEMORY                                0x00000001
#define LWE22D_G2CONTROLMAIN_0_DST_RD_WR_SEP                                29:29
#define LWE22D_G2CONTROLMAIN_0_DSTDIR                                       28:28
#define LWE22D_G2CONTROLMAIN_0_SRCDIR                                       27:27
#define LWE22D_G2CONTROLMAIN_0_GCSW                                         26:25
#define LWE22D_G2CONTROLMAIN_0_SRCBAS                                       24:24
#define LWE22D_G2CONTROLMAIN_0_SRCT                                         23:22
#define LWE22D_G2CONTROLMAIN_0_HLMONO                                       21:21
#define LWE22D_G2CONTROLMAIN_0_SRCCD                                        20:20
#define LWE22D_G2CONTrOLMAIN_0_DSTT                                         19:18
#define LWE22D_G2CONTROLMAIN_0_DSTCD                                        17:16
#define LWE22D_G2CONTROLMAIN_0_DSTCD_BPP8                                   0x00000000
#define LWE22D_G2CONTROLMAIN_0_DSTCD_BPP16                                  0x00000001
#define LWE22D_G2CONTROLMAIN_0_DSTCD_BPP32                                  0x00000002
#define LWE22D_G2CONTROLMAIN_0_DSTCD_RESERVED3                              0x00000003
#define LWE22D_G2CONTROLMAIN_0_SRCSEL                                       15:15
#define LWE22D_G2CONTROLMAIN_0_SRCSEL_SCREEN                                0x00000000
#define LWE22D_G2CONTROLMAIN_0_SRCSEL_MEMORY                                0x00000001
#define LWE22D_G2CONTROLMAIN_0_YFLIP                                        14:14
#define LWE22D_G2CONTROLMAIN_0_YFLIP_DSIABLE                                0x00000000
#define LWE22D_G2CONTROLMAIN_0_YFLIP_ENABLE                                 0x00000001
#define LWE22D_G2CONTROLMAIN_0_PATPACK                                      13:13
#define LWE22D_G2CONTROLMAIN_0_PATPACK_DISABLE                              0x00000000
#define LWE22D_G2CONTROLMAIN_0_PATPACK_ENABLE                               0x00000001
#define LWE22D_G2CONTROLMAIN_0_SRCPACK                                      12:12
#define LWE22D_G2CONTROLMAIN_0_SRCPACK_DISABLE                              0x00000000
#define LWE22D_G2CONTROLMAIN_0_SRCPACK_ENABLE                               0x00000001
#define LWE22D_G2CONTROLMAIN_0_XYTDW                                        11:11
#define LWE22D_G2CONTROLMAIN_0_YDIR                                         10:10
#define LWE22D_G2CONTROLMAIN_0_XDIR                                         9:9
#define LWE22D_G2CONTROLMAIN_0_PATFL                                        8:8
#define LWE22D_G2CONTROLMAIN_0_PATFL_DISABLE                                0x00000000
#define LWE22D_G2CONTROLMAIN_0_PATFL_ENABLE                                 0x00000001
#define LWE22D_G2CONTROLMAIN_0_PATSLD_DISABLE                               0x00000000
#define LWE22D_G2CONTROLMAIN_0_PATSLD_ENABLE                                0x00000001
#define LWE22D_G2CONTROLMAIN_0_PATSLD                                       7:7
#define LWE22D_G2CONTROLMAIN_0_SRCSLD                                       6:6
#define LWE22D_G2CONTROLMAIN_0_SRCSLD_DISABLE                               0x00000000
#define LWE22D_G2CONTROLMAIN_0_SRCSLD_ENABLE                                0x00000001
#define LWE22D_G2CONTROLMAIN_0_ALPEN                                        5:5
#define LWE22D_G2CONTROLMAIN_0_FADEN                                        4:4
#define LWE22D_G2CONTROLMAIN_0_FADEN_DISABLE                                0x00000000
#define LWE22D_G2CONTROLMAIN_0_FADEN_ENABLE                                 0x00000001
#define LWE22D_G2CONTROLMAIN_0_TEST0BIT                                     3:3
#define LWE22D_G2CONTROLMAIN_0_TEST0BIT_DISABLE                             0x00000000
#define LWE22D_G2CONTROLMAIN_0_TEST0BIT_ENABLE                              0x00000001
#define LWE22D_G2CONTROLMAIN_0_TURBOFILL                                    2:2
#define LWE22D_G2CONTROLMAIN_0_CMDT                                         1:0
#define LWE22D_G2CONTROLMAIN_0_CMDT_BITBLT                                  0x00000000
#define LWE22D_G2CONTROLMAIN_0_CMDT_LINEDRAW                                0x00000001
#define LWE22D_G2CONTROLMAIN_0_CMDT_VCAA                                    0x00000002
#define LWE22D_G2CONTROLMAIN_0_CMDT_RESERVED1                               0x00000003


/*
 *  ROP:
 *  If YFLIP==1 or XYTDW==1, ROP can not include destination.
 *  Since destination may have been corrupted before reading out.
 */
#define LWE22D_G2ROPFADE_0                                                  (0x20)
#define LWE22D_G2ROPFADE_0_FADOFF                                           31:24
#define LWE22D_G2ROPFADE_0_FADCOE                                           23:16
#define LWE22D_G2ROPFADE_0_ROP                                              7:0


#define LWE22D_G2ALPHABLEND_0                                               (0x21)
#define LWE22D_G2ALPHABLEND_0_ALPHAIV                                       31:24
#define LWE22D_G2ALPHABLEND_0_ALPHAOV                                       23:16
#define LWE22D_G2ALPHABLEND_0_ALPHAILW                                      8:8
#define LWE22D_G2ALPHABLEND_0_ALPHA                                         7:0


#define LWE22D_G2CLIPLEFTTOP_0                                              (0x22)
#define LWE22D_G2CLIPLEFTTOP_0_CLIPT                                        30:16
#define LWE22D_G2CLIPLEFTTOP_0_CLIPL                                        14:0


#define LWE22D_G2CLIPRIGHTBOT_0                                             (0x23)
#define LWE22D_G2CLIPRIGHTBOT_0_CLIPB                                       30:16
#define LWE22D_G2CLIPRIGHTBOT_0_CLIPR                                       14:0

/*
 *  pattern methods
 *   G2PATPACK should be used to specify the line gap
 *   use G2PATPACK_SIZE to program height and width
 *    PACK is generally only useful with narrow monochrome surfaces
 *    Note, PACK is not offically supported when the surface is tiled.
 */
#define LWE22D_G2PATPACK_0                                                  (0x24)
#define LWE22D_G2PATPACK_0_PATLNGAP                                         3:0

/*
 *   G2PATPACK_SIZE -- extension of G2PATPACK, this register only holds the size of the packed data
 *    Note, PACK is not offically supported when the surface is tiled.

 *   Register G2SB_G2PATPACK_SIZE_0  //{2d,index=25} Pattern packed mode
 *   Packed mode, pattern data line gap. byte
 */
#define LWE22D_G2PATPACK_SIZE_0                                             (0x25)
#define LWE22D_G2PATPACK_SIZE_0_PATMOHOH                                    31:16
#define LWE22D_G2PATPACK_SIZE_0_PATMONOW                                    15:0


/*
 *   If(PATFL==1){
 *     It has to be 16bytes aligned.
 *   }else{
 *     Point to the first byte of the first pixel of pattern plane.
 *   }
 */
#define LWE22D_G2PATBA_0                                                    (0x26)
#define LWE22D_G2PATBA_0_PATBA                                              31:0


/*
 *  PATY0
 *  y offset for mono tile pattern fill.  see PATFL
 *  PATX0
 *  x offset for mono tile pattern fill.  see PATFL
 *   PATT:
 *   Mono pattern transparency enable: 
 *   0x=pattern transparency disabled, 
 *   10=mono pattern background transparency 
 *     or color pattern transparency, 
 *   11=mono pattern foreground transparency 
 *     or ilwerse color pattern transparency.
 *   NOTE: When pattern transparency is enabled and PATCD==0(mono)
 *         PATBGC!=PATFGC should be satisfied.
 *  PATCD:
 *  0 mono
 *  1 same as dstcd
 *    PATCD==0 xdir/ydir has to be 0
 */
#define LWE22D_G2PATOS_0                                                    (0x27)
#define LWE22D_G2PATOS_0_PATY0                                              31:28
#define LWE22D_G2PATOS_0_PATX0                                              27:24
#define LWE22D_G2PATOS_0_PATT                                               22:21
#define LWE22D_G2PATOS_0_PATCD                                              16:16
#define LWE22D_G2PATOS_0_PATST                                              15:0



#define LWE22D_G2PATBGC_0                                                   (0x28)
#define LWE22D_G2PATBGC_0_PATBGC                                            31:0


#define LWE22D_G2PATFGC_0                                                   (0x29)
#define LWE22D_G2PATFGC_0_PATFGC                                            31:0


#define LWE22D_G2PATKEY_0                                                   (0x2a)
#define LWE22D_G2PATKEY_0_PATKEY                                            31:0

#define LWE22D_G2DSTBA_0                                                    (0x2b)
#define LWE22D_G2DSTBA_0_DSTBA                                              31:0

#define LWE22D_G2DSTBA_B_0                                                  (0x2c)
#define LWE22D_G2DSTBA_B_0_DSTBA_B                                          31:0

#define LWE22D_G2DSTBA_C_0                                                  (0x2d)
#define LWE22D_G2DSTBA_C_0_DSTBA_C                                          31:0

/*
 *   Destination Stride coordinate(bytes) with respect to DSTBA.
 */
#define LWE22D_G2DSTST_0                                                    (0x2e)
#define LWE22D_G2DSTST_0_DSTS                                               15:0


/*
 *   Surface methods
 *   G2SRCPACK should only be used to specify the line gap
 *   use G2SRCPACK_SIZE to program height and width
 *   Packed mode - source mono data line gap
 */
#define LWE22D_G2SRCPACK_0                                                  (0x2f)
#define LWE22D_G2SRCPACK_0_SRCLNGAP                                         3:0


/*
 *   G2SRCPACKS_SIZE -- extension of G2SRCPACK, this register only holds the size of the packed data
 *
 *   Register G2SB_G2SRCPACK_SIZE_0  //{2d,index=30} source data packed mode
 *   In packed mode, SRCMONOW holds the horizontal size (bytes).
 *   If MONOH > 1, it is required width/stride be 16 byte aligned.
 *   Packed mode - source mono data line gap
 */
#define LWE22D_G2SRCPACK_SIZE_0                                             (0x30)
#define LWE22D_G2SRCPACK_SIZE_0_SRCMONOH                                    31:16
#define LWE22D_G2SRCPACK_SIZE_0_SRCMONOW                                    15:0


/*
 *   Source base address (byte address) 
 */
#define LWE22D_G2SRCBA_0                                                    (0x31)
#define LWE22D_G2SRCBA_0_SRCBA                                              31:0

/*
 *  This parameter specifies the start address of source image 
 *   stored in the image buffer memory. In 4:2:0 
 *  format mode, this image block accommodates for Y-image.
 *  This address specifies byte-position, however, bits [2:0] 
 *   are restricted with respect to the data formats to fit 
 *  multiple pixels in one memory word (8 bytes),. For example, 
 *   {0, 4} for any YcrCb formats, {0, 2, 4, 6} for 
 *  RGB 16-bit format.  Since one Y pixel takes 8-bit, all 
 *   8 byte-positions are valid in 4:2:0 mode. (Unlike 
 *  multiplexed pixels format, there is no restrictions on this value.)
 */
#define LWE22D_G2SRCBA_0_B                                                  (0x32)
#define LWE22D_G2SRCBA_0_SRCBA_B                                            31:0


/*
 *   Source Stride coordinate(bytes) with respect to SRCBA.
 *  In order to fit multiple pixels in one memory word (8 bytes),
 *    bits [2:0] are restricted with respect to the 
 *  data formats. For example, {0, 4} for any YcrCb formats,
 *    {0, 2, 4, 6} for RGB 16-bit format. 
 */
#define LWE22D_G2SRCST_0                                                    (0x33)
#define LWE22D_G2SRCST_0_SRCS                                               15:0


#define LWE22D_G2SRCBGC_0                                                   (0x34)
#define LWE22D_G2SRCBGC_0_SRCBGC                                            31:0


#define LWE22D_G2SRCFGC_0                                                   (0x35)
#define LWE22D_G2SRCFGC_0_SRCFGC                                            31:0


#define LWE22D_G2SRCKEY_0                                                   (0x36)
#define LWE22D_G2SRCKEY_0_SRCKEY                                            31:0

/*
 *   SRCHEIGHT:
 *   In SB mode, number of lines - 1
 *   In 2D mode, actual lines
 */
#define LWE22D_G2SRCSIZE_0                                                  (0x37)
#define LWE22D_G2SRCSIZE_0_SRCHEIGHT                                        30:16
#define LWE22D_G2SRCSIZE_0_SRCWIDTH                                         14:0


/*
 *   DSTHEIGHT
 *   In SB mode, number of lines - 1
 *   In 2D mode, actual lines
 */
#define LWE22D_G2DSTSIZE_0                                                  (0x38)
#define LWE22D_G2DSTSIZE_0_DSTHEIGHT                                        30:16
#define LWE22D_G2DSTSIZE_0_DSTWIDTH                                         14:0

/*
 *   ImageBlit Methods
 *
 *  SRCX[2:0] are ignored in SRCCD==0 (mono expansion), 
 *  The first bit of the first byte (bit7 if HLMONO==0, or bit0 if HLMONO==1) always
 *  expand to DSTX,DSTY  
 */
#define LWE22D_G2SRCPS_0                                                    (0x39)
#define LWE22D_G2SRCPS_0_SRCY                                               31:16
#define LWE22D_G2SRCPS_0_SRCX                                               15:0

/*
 *   NOTE: when ALPTYPE is PL1BPP, DSTX[2:0] must be zero, see bug 344489
 */
#define LWE22D_G2DSTPS_0                                                    (0x3a)
#define LWE22D_G2DSTPS_0_DSTY                                               31:16
#define LWE22D_G2DSTPS_0_DSTX                                               15:0


/*
 *   cirlwlar buffer 
 *  
 *  CBLINE
 *  vertical line number in one buffer
 *   CBCOUNT:
 *   This specifies the number of buffers in
 *  cirlwlar buffer feature                   
 */
#define LWE22D_G2CBDES_0                                                    (0x3b)
#define LWE22D_G2CBDES_0_TOPCLIP                                            31:31
#define LWE22D_G2CBDES_0_TOPCLIP_DISABLE                                    0x00000000
#define LWE22D_G2CBDES_0_TOPCLIP_ENABLE                                     0x00000001
#define LWE22D_G2CBDES_0_CBLINE                                             30:16
#define LWE22D_G2CBDES_0_CBCOUNT                                            7:0

/*
 *   CBUVSTRIDE
 *   Chroma Buffer Stride default is half of luma
 *     00= 1/2 of Luma Buffer Stride; in this
 *         case, Luma Buffer Stride should be
 *         multiple of 2 bytes.
 *     01= Equal to Luma Buffer Stride
 *     10= 1/4 of Luma Buffer Stride; in this
 *         case, Luma Buffer Stride should be
 *         multiple of 4 bytes.
 *     1x= Reserved
 *   CBSTRIDE
 *   Video Buffer Luma(or RGB) Buffer Stride
 *    This is luma buffer stride (in bytes)
 */
#define LWE22D_G2CBSTRIDE_0                                                 (0x3c)
#define LWE22D_G2CBSTRIDE_0_CBUVSTRIDE                                      31:30
#define LWE22D_G2CBSTRIDE_0_CBUVSTRIDE_CBS2X                                0x00000000
#define LWE22D_G2CBSTRIDE_0_CBUVSTRIDE_CBS1X                                0x00000001
#define LWE22D_G2CBSTRIDE_0_CBUVSTRIDE_CBS4X                                0x00000002
#define LWE22D_G2CBSTRIDE_0_CBSTRIDE                                        23:0


/*
 *  Line Methods
 *  
 *  OCTANTS
 *  000 octant 0
 *  001 octant 1
 *  010 octant 2
 *  011 octant 3
 *  100 octant 4
 *  101 octant 5
 *  110 octant 6
 *  111 octant 7
 *  LINEUSEOCTANT:
 *  use OCTANTS in G2LINEDELTAN register instead of MAJOR LINEXDIR LINEYDIR
 *  DROPLASTP:
 *  draw last pixel or not
 *  MAJOR:
 *  0:xmajor 1: y major
 */
#define LWE22D_G2LINESETTINGS_0                                             (0x3d)
#define LWE22D_G2LINESETTINGS_0_OCTANTS                                     31:29
#define LWE22D_G2LINESETTINGS_0_LINEUSEOCTANT                               28:28
#define LWE22D_G2LINESETTINGS_0_DROPLASTP                                   27:27
#define LWE22D_G2LINESETTINGS_0_LINEYDIR                                    26:26
#define LWE22D_G2LINESETTINGS_0_LINEXDIR                                    25:25
#define LWE22D_G2LINESETTINGS_0_MAJOR                                       24:24
#define LWE22D_G2LINESETTINGS_0_GAMMA                                       20:0


/*
 *   
 *   
 *     |    |    | 
 *      | 5 | 6 | 
 *       |  |  | 
 *      4 | | | 7
 *         ||| 
 *     ----------->
 *         ||| 
 *      3 | | | 0
 *       |  |  | 
 *      | 2 | 1 | 
 *     |    |    | 
 *          V
 *  
 */

#define LWE22D_G2LINEDELTAN_0                                               (0x3e)
#define LWE22D_G2LINEDELTAN_0_DELTAN                                        20:0

#define LWE22D_G2LINEDELTAM_0                                               (0x3f)
#define LWE22D_G2LINEDELTAM_0_DELTAM                                        20:0

#define LWE22D_G2LINEPOS_0                                                  (0x40)
#define LWE22D_G2LINEPOS_0_LINEYPOS                                         31:16
#define LWE22D_G2LINEPOS_0_LINEXPOS                                         15:0

#define LWE22D_G2LINELEN_0                                                  (0x41)
#define LWE22D_G2LINELEN_0_LINELEN                                          14:0

/*
 *   G2V
 *   multiplier for G for V generation. 
 *  This parameter consists of a sign bit and 8-bit magnitude (s1.7)
 *   For RGB->YUV the recommended value is -0.368 (decimal) or 0x12F
 *   For any other combination this parameter is ignored
 *   G2U:
 *   multiplier for G for U generation.
 *  This parameter consists of a sign bit and 8-bit magnitude (s1.7)
 *   For RGB->YUV the recommended value is -0.291 (decimal) or 0x125
 *   For any other combination this parameter is ignored
 */
#define LWE22D_G2CSCFOURTH_0                                                (0x42)
#define LWE22D_G2CSCFOURTH_0_G2V                                            24:16
#define LWE22D_G2CSCFOURTH_0_G2U                                            8:0


/*
 *   SRCS_B:
 *   Source Stride B
 */
#define LWE22D_G2SRCST_B_0                                                  (0x43)
#define LWE22D_G2SRCST_B_0_SRCS_B                                           15:0

#define LWE22D_G2UVSTRIDE_0                                                 (0x44)
#define LWE22D_G2UVSTRIDE_0_UVSTRIDE                                        15:0

/*
 *   cirlwlar buffer controller 2
 *  
 *   CBLINE:
 *   Cirlwlar buffer top clipping enabled, the first buffer line num
 */
#define LWE22D_G2CBDES2_0                                                   (0x45)
#define LWE22D_G2CBDES2_0_CBLINE                                            14:0

/*
 *   DST_RD_TILE_MODE:
 *  Same as destination write unless DST_RD_WR_SEP (not supported)
 *   PAT_UV_TILE_MODE: UNUSED
 *  SRC_UV_TILE_MODE: UV surface, ignored in RGB mode
 *  SRC_Y_TILE_MODE: Y or RGB surface
 */
#define LWE22D_G2TILEMODE_0                                                 (0x46)
#define LWE22D_G2TILEMODE_0_DST_WR_TILE_MODE                                20:20
#define LWE22D_G2TILEMODE_0_DST_WR_TILE_MODE_LINEAR                         0x00000000
#define LWE22D_G2TILEMODE_0_DST_WR_TILE_MODE_TILED                          0x00000001
#define LWE22D_G2TILEMODE_0_DST_RD_TILE_MODE                                16:16
#define LWE22D_G2TILEMODE_0_DST_RD_TILE_MODE_LINEAR                         0x00000000
#define LWE22D_G2TILEMODE_0_DST_RD_TILE_MODE_TILED                          0x00000001
#define LWE22D_G2TILEMODE_0_PAT_UV_TILE_MODE                                12:12
#define LWE22D_G2TILEMODE_0_PAT_UV_TILE_MODE_LINEAR                         0x00000000
#define LWE22D_G2TILEMODE_0_PAT_UV_TILE_MODE_TILED                          0x00000001
#define LWE22D_G2TILEMODE_0_PAT_Y_TILE_MODE                                 8:8
#define LWE22D_G2TILEMODE_0_PAT_Y_TILE_MODE_LINEAR                          0x00000000
#define LWE22D_G2TILEMODE_0_PAT_Y_TILE_MODE_TILED                           0x00000001
#define LWE22D_G2TILEMODE_0_SRC_UV_TILE_MODE                                4:4
#define LWE22D_G2TILEMODE_0_SRC_UV_TILE_MODE_LINEAR                         0x00000000
#define LWE22D_G2TILEMODE_0_SRC_UV_TILE_MODE_TILED                          0x00000001
#define LWE22D_G2TILEMODE_0_SRC_Y_TILE_MODE                                 0:0
#define LWE22D_G2TILEMODE_0_SRC_Y_TILE_MODE_LINEAR                          0x00000000
#define LWE22D_G2TILEMODE_0_SRC_Y_TILE_MODE_TILED                           0x00000001


/*
 *  pattern base address in tile mode, 
 *   PATBA is the linear address where pixel start
 */
#define LWE22D_G2PATBASE_0                                                  (0x47)
#define LWE22D_G2PATBASE_0_PAT_BASE                                         31:0


/*
 *   SB_SURFBASE registers
 *      These registers need only be programmed when using SB.  They point to the base
 *          of the various source and destination surfaces.  Technically they are only needed when 
 *          tiling is enabled, but there is no harm in always programming them.   
 *      Their counterpart registers (<reg>_SB_SURFBASE with SB_SURFBASE stripped off) indicate the 
 *          location of the first pixel  to be sourced or written within the surface.
 *   
 *      This register exists to mimic the X, Y (SRCX, SRCY) functionality found in the BitBlt engine.
 *      
 *      For example, to get at pixel X, Y:
 *          SRCBA = SRCBA_SB_SURFBASE + Y*stride + X*Bpp
 *   surface address corresponding to G2SRCBA:
 *      -base of interleaved sources (RGB, YUV)
 *      -base of Y plane
 *   Only used by the StretchBlit Engine
 */

#define LWE22D_G2SRCBA_SB_SURFBASE_0                                        (0x48)
#define LWE22D_G2SRCBA_SB_SURFBASE_0_SRC_ADDR                               31:0


/*
 *   surface address corresponding to G2DSTBA
 *   Only used by the StretchBlit Engine
 */
#define LWE22D_G2DSTBA_SB_SURFBASE_0                                        (0x49)
#define LWE22D_G2DSTBA_SB_SURFBASE_0_DST_ADDR                               31:0


/*
 *   surface address corresponding to G2DSTBA_B
 *   Only used by the StretchBlit Engine, and G2CONTROLSB.DBSEL() is enabled
 */
#define LWE22D_G2DSTBA_B_SB_SURFBASE_0                                      (0x4a)
#define LWE22D_G2DSTBA_B_SB_SURFBASE_0_DST_B_ADDR                           31:0


/*
 *   surface address corresponding to G2VBA
 *       used for YUV 4:2:0 planar, base of V plane
 *   Only used by the StretchBlit Engine
 */
#define LWE22D_G2VBA_A_SB_SURFBASE_0                                        (0x4b)
#define LWE22D_G2VBA_A_SB_SURFBASE_0_V_ADDR                                 31:0


/*
 *   surface address corresponding to G2VBA
 *       used for YUV 4:2:0 planar, base of U plane
 *   Only used by the StretchBlit Engine
 */
#define LWE22D_G2UBA_A_SB_SURFBASE_0                                        (0x4c)
#define LWE22D_G2UBA_A_SB_SURFBASE_0_U_ADDR                                 31:0

#endif // ifndef ___ARG2SB_H_INC_
