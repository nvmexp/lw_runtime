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
#ifndef _cl_e2b0_h_
#define _cl_e2b0_h_

#include "lwtypes.h"
#define LW_E2_ISP                       (0xE2B0)

/*
 * ISP (Image Signal Processor) register definition
 *
 * The ISP takes digital image signal from CMOS imagers and colwerts to YUV444 signal.
 *
 * Colwentional CMOS/CCD imagers (or sensors) are built with Bayer color filter array (CFA) 
 * and their output image signal is an alternating series of image lines, one with interleaved 
 * R and G, and another one with interleaved G and B. 
 * Another type of CMOS imagers, namely stacked pixel type, shall be accepted by the ISP in the
 * future. The stacked pixel type imagers have three layers of sensors for R, G, anf B stucked
 * on top of others, and are capable of producing three color signals at every pixel position
 * in the imaging area.
 * 
 * An imager that supplies pixel stream to the ISP is the master of timing, and also supplies
 * pixel clock, H and V sync timing pulses.
 * The ISP is a slave to the imager, and locally regenerates line and frame timing based on the
 * H and V input sync pulses.
 * A variety of imagers from CIF size to 5+ mega-pixel are acceptable. Overall frame timing is
 * defined by the input V and H pulses. Within the frame period, positions of optical black
 * lines at the top and active image lines are specified by the host through control parameter
 * registers.
 * Likewise, positions of optical black pixels at the left and right, and active image pixels 
 * within the line period are specified through control parameter registers.
 * 
 * In AP15, The ISP receives sensor's raw data via "VI2ISP" data bus.  VI2ISP is a SVD bus
 * (please refer arvideo.spec for the definition of SVD),  The bayer data is on the lower 16 bits
 * of the 24-bit VI2ISP bus, and it's in the format called "BAYER3D".  BAYER3D assumes the 
 * raw data has 14 effective bits, and it puts 7 bits on the LSBs of each of the two bytes.
 * ISP needs to grab the 10 "effective" most significant bits from VI2ISP bus to feed into its
 * data processing pipe.  The following shows where the 10 "effective" MSBs are:
 *
 * vi2isp bits    fedcba98 76543210
 * isp receives   -dcba987 -6543210
 * isp takes:     -dcba987 -654----
 * isp names:     -9876543 -210----
 *  
 * The ISP outputs processed pixel data in YUV444 format.  The output bus "ISP2VI" is also 
 * a SVD bus.    
 * The sensor's output data pins need to be aligned with VI's input pins at MSB to ensure  
 * ISP's function correctness. 
 * In bypass mode (demosaic_enable=0), ISP just forwards the data on vi2isp bus to isp2vi bus
 * without any change.
 *                22221111 111111
 * isp2vi bits    32109876 54321098 76543210
 * isp received:  -------- -dcba987 -6543210
 * isp sends:     *dcba987 *dcba987 *6543210
 * 
 * Note that * means "0" (low level).
 * Note that bits [23:16] carry a copy of bits [15:8].
 *
 * There are 3 kinds of output images selectable with "DEMOSAIC_ENABLE" and "OUT_IMAGE_TYPE".
 * Notice that "DEMOSAIC_ENABLE" means "pixel-processing-enable" and not narrow sense of 
 *   enable for the demosaic (color-interpolation) section.
 *   "DEMOSAIC_ENABLE" "OUT_IMAGE_TYPE"     kind of output image
 *   --------------------------------------------------------------
 *         0                0               input Bayer image (bypassed)
 *         0                1               input Bayer image (bypassed)
 *         1                0               demosaiced YUV or RGB image
 *         1                1               processed Bayer image
 *  
 *
 * Using 2D interpolation, input image frames in R/G/B mosaic pattern are demosaiced and
 * colwerted to Y/U/V format.
 * A number of line delay buffers are used in the demosaicing. For example, 2 line-buffers are
 * used for the simplest configuration, and 2 groups of line buffers of 4-line and 2-line each 
 * are used for the preferred configuration.
 * These line buffers are allocated in the display memory. Multiple write and read ports 
 * logically required for the line buffers are to be mapped to physical memory of 1
 * bidirectional port.
 * A stream of pixels (10-bit per pixel for example) are parallelized to fit into 128 bits of
 * a memory word, and appropriately interleaved for the most efficient memory access operation.
 *
 * There are a number of operation modes for demosaicing and related pixel processing, in which
 * 2D datapath, demosaicing method, image enhancement processing and so on are pre-arranged.
 * By selecting one operation mode, all the related elements are setup as pre-arranged.
 * 
 * The first operation to be applied to the input pixel signal is to establish "black" level.
 * This is to anchor signal activities to the black reference level, or "zero" level. Optical
 * black pixels, which are arranged to surround the active imaging area, are used to determine
 * the input signal level of reference black, which is mapped to "zero" level in the following
 * pixel processing. CMOS imagers tend to show a small fractuation of black level by the column 
 * position along every image line, so "fixed column noise pattern" will be axtracted and then
 * used for cancellation. The input signal level of reference black may vary line by line within
 * an image frame, so the optical black pixels on every line are referred to to detect and 
 * compensate the fluctuations.
 * Stacked pixel type imagers do not have fixed column noise, but they show cross-talk 
 * (or leakagea) of one pixel to the next one, due to their readout circuit arrangement. So 
 * there must be a "column filter" that reverses the cross-talk.
 *
 * Imagers tend to show some non-linearity in their light-to-electric transfer characteristics.
 * There is a logical LUT provided to linearize the transfer characteristics, which is
 * implemented in a multi-segment piecewise linear line generator. 
 *
 * The optical lens placed in front of the imager show some degree of shading. Cheaper lens
 * tends to show larger shading. To compensate the lens shading, a varying "gain factor" as a
 * function of location in the image frame is multiplied to ever pixel. 
 *
 * Two blocks of memory space are reserved for ISP
 *   1. Sampled pixel values result from M3 statistics gathering (see M3_MEM_START_ADDR)
 *   2. Sampled pixel values for detection of flicker bands (see FB_MEM_START_ADDR)
 * 
 */

// Memory address width
// Input resolution
// Horizontal input resolution (# of bits)
#define LWE2B0_H_IN     14
// Vertical   input resolution (# of bits)
#define LWE2B0_V_IN     14
// Width or height of optical black (# of bits)
#define LWE2B0_OB_IN    4
// Number of sensor pixels (# of bits)
#define LWE2B0_PIXEL_IN 25
// Number of bits per "BAYER3D" pixels
#define LWE2B0_PIXEL_BAYER3D    14
// Memory size of the Fixed Column Noise pattern
// Number of words (pixels)
#define LWE2B0_FCN_MEM_SIZE     1200
// Pixel input from imagers
// Sensor data bus width
#define LWE2B0_SENSOR_DIBIT     12
// Bayer CFA type imager (number of bits/pixel)
#define LWE2B0_BAYER_DIBIT      10
#define LWE2B0_BAYER_DIMAX      1023
// Pixel stack type imager (number of bits/pixel)
#define LWE2B0_STACK_DIBIT      12
// Chose a type of imager
#define LWE2B0_PIXEL_DIMAX      1023
#define LWE2B0_PIXEL_DIBIT      10
// Pixel resolution in gain/white balance and demosaic
// Number of bits per pixel
#define LWE2B0_DEM_DPBIT        10
// Optical Black and Fixed Column Noise Processing
// OB acquisition, bit width of filter coefficient (coded)
#define LWE2B0_OB_KBIT  4
// OB processing (number of bits/pixel)
// ISP_OB_DBIT is obsolete and to be removed in next project, 
// refer to ISP_OBR_DBIT instead.
#define LWE2B0_OB_DBIT  6
#define LWE2B0_OBR_DBIT 10
// OB filter, fraction bits (number of bits)
#define LWE2B0_OB_FBIT  5
// Additional bits for LOB callwlation
#define LWE2B0_LOB_ADD  3
// LOB column width 
#define LWE2B0_LOB_WIDTH        2
// FCN acquisition, bit width of filter coefficient (coded)
#define LWE2B0_FCN_KBIT 4
// Column filter, bit width of coefficient (coded)
#define LWE2B0_CF_KBIT  4
// Common gain applied after OB restoration
#define LWE2B0_CG_DBIT  10
#define LWE2B0_CG_FBIT  5
// De-Knee, Input Linearization Processing
// This is implemented as piecewise linear approximation of equally spaced segments
// Maximum number of line segments for the piecewise linear approximation
#define LWE2B0_DEKNEE_NSEG      16
// Number of bits for input and output
#define LWE2B0_DEKNEE_DBIT      10
// U2.10, slope value of a de-knee line segment
#define LWE2B0_DEKNEE_SBIT      12
// Nubmer of fraction bits for the de-knee line segment slope values
#define LWE2B0_DEKNEE_FBIT      10
// Lens Shading Compensation Processing (legacy)
// U3.6  coefficient of 2nd-order term
#define LWE2B0_SHADING_K2BIT    9
// S2.6  coefficient of 1st-order term, signed magnitude (planned)
// U3.6  positive only (actual)
#define LWE2B0_SHADING_K1BIT    9
// Lens Shading Compensation Processing (new)
// Number of control points per patch (row and column)
#define LWE2B0_LS_CTRL_PT_PER_PATCH_COLUMN      4
#define LWE2B0_LS_CTRL_PT_PER_PATCH_ROW 4
// Number of patches per image row and column
#define LWE2B0_LS_PATCH_PER_COLUMN      3
#define LWE2B0_LS_PATCH_PER_ROW 3
// Number of control points per image row and column 
#define LWE2B0_LS_CTRL_PT_PER_COLUMN    10
#define LWE2B0_LS_CTRL_PT_PER_ROW       10
// Number of stored control points total (for all 4 colors)
#define LWE2B0_LS_NUM_ST_CTRL_PT        480
#define LWE2B0_LS_NUM_ST_CTRL_PT_LOG2   9
// S4.13  Control points for Bezier patch surfaces
#define LWE2B0_LS_SIGN_CBIT     1
#define LWE2B0_LS_INT_CBIT      4
#define LWE2B0_LS_FRAC_CBIT     13
#define LWE2B0_LS_CBIT  18
// U1.34  1 / ( PATCH_WIDTH/2 ) where PATCH_WIDTH >= 12 and is even
#define LWE2B0_LS_SIGN_DELTAUBIT        0
#define LWE2B0_LS_INT_DELTAUBIT 1
#define LWE2B0_LS_FRAC_DELTAUBIT        34
#define LWE2B0_LS_DELTAUBIT     35
// U1.34  1 / ( PATCH_HEIGHT/2 ) where PATCH_HEIGHT >= 4 and is even
#define LWE2B0_LS_SIGN_DELTAVBIT        0
#define LWE2B0_LS_INT_DELTAVBIT 1
#define LWE2B0_LS_FRAC_DELTAVBIT        34
#define LWE2B0_LS_DELTAVBIT     35
// U1.22  Normalized horizontal coordinate (aclwmulation of delta u)
#define LWE2B0_LS_SIGN_UBIT     0
#define LWE2B0_LS_INT_UBIT      1
#define LWE2B0_LS_FRAC_UBIT     22
#define LWE2B0_LS_UBIT  23
// U1.22  Normalized horizontal coordinate (aclwmulation of delta v)
#define LWE2B0_LS_SIGN_VBIT     0
#define LWE2B0_LS_INT_VBIT      1
#define LWE2B0_LS_FRAC_VBIT     22
#define LWE2B0_LS_VBIT  23
// S5.17  Precision of mul in vertical interpolation
//        Truncated ( ( S4.13 - S4.13 ) * U1.22 = S5.13 * U1.22 = S5.25 )
//        If we assume U1.22 <= 1.0, prod should be max S5.25
#define LWE2B0_LS_SIGN_V_PRODBIT        1
#define LWE2B0_LS_INT_V_PRODBIT 5
#define LWE2B0_LS_FRAC_V_PRODBIT        17
#define LWE2B0_LS_V_PRODBIT     23
// S4.17  Precision of add in vertical interpolation
//        Truncated S5.17 + S4.13 = S4.17
//        If we assume U1.22 is max 1.0, sum is the average
//        so should be max S4.17
#define LWE2B0_LS_SIGN_V_SUMBIT 1
#define LWE2B0_LS_INT_V_SUMBIT  4
#define LWE2B0_LS_FRAC_V_SUMBIT 17
#define LWE2B0_LS_V_SUMBIT      22
// S5.16  Precision of mul in horizontal interpolation
//        Truncated ( ( S4.13 - S4.13 ) * U1.22 = S5.13 * U1.22 = S5.25 )
//        If we assume U1.22 <= 1.0, prod should be max S5.25
#define LWE2B0_LS_SIGN_H_PRODBIT        1
#define LWE2B0_LS_INT_H_PRODBIT 5
#define LWE2B0_LS_FRAC_H_PRODBIT        16
#define LWE2B0_LS_H_PRODBIT     22
// S4.16  Precision of add in horizontal interpolation
//        Truncated S5.16 + S4.13 = S4.16
//        If we assume U1.22 is max 1.0, sum is the average
//        so should be max S4.16
#define LWE2B0_LS_SIGN_H_SUMBIT 1
#define LWE2B0_LS_INT_H_SUMBIT  4
#define LWE2B0_LS_FRAC_H_SUMBIT 16
#define LWE2B0_LS_H_SUMBIT      21
// U4.12  Final lens shading gain
#define LWE2B0_LS_SIGN_QBIT     0
#define LWE2B0_LS_INT_QBIT      4
#define LWE2B0_LS_FRAC_QBIT     12
#define LWE2B0_LS_QBIT  16
// Gain and White Balance 
// # of fraction bits for the temporal filters
#define LWE2B0_WB_FBIT  6
// U3.7  balancing gain factor
//   total bit count of KBIT
//   including KFBIT bits of fraction part
//#define LWE2B0_WB_KBIT           10
#define LWE2B0_WB_KBIT  10
#define LWE2B0_WB_KFBIT 7
// Bit length of frequency counter
#define LWE2B0_WB_HBIT  10
// Selection of temporal filter coefficients (coded)
#define LWE2B0_WB_FKBIT 3
// Max limit for the peak detect (MSP 8-bit), 8'b1111_1100 
#define LWE2B0_WB_SMAX8 252
// Bit length of sample counter
#define LWE2B0_WB_SPLBIT        6
// Bad pixel
#define LWE2B0_BADPIXEL_LT_FBITS        3
#define LWE2B0_BADPIXEL_UT_FBITS        3
#define LWE2B0_BADPIXEL_LT_CBITS        4
#define LWE2B0_BADPIXEL_UT_CBITS        4
// Edge Enhancement and noise reduction
// Number of bits to specify coring threshold level
#define LWE2B0_CORE_DBIT        6
// Number of bits to represent coefficient for edge enhancement
#define LWE2B0_EENHANCE_CBIT    5
// Number of fraction bits used by the coefficient              
#define LWE2B0_EENHANCE_FBIT    3
// Color Correction
// Number of bits for positive only (unsigned) factors
#define LWE2B0_CC_PKBIT 11
// Number of bits for pos/neg (signed) factors
#define LWE2B0_CC_NKBIT 12
// Number of bits for fraction for the coefficients
#define LWE2B0_CC_FBIT  8
// Number of bits for fraction for the products
#define LWE2B0_CC_FPBIT 4
// Gamma Correction
// This is implemented as piecewise linear approximation of arbitrarily spaced segments
// Number of line segments for the piecewise linear approximation
#define LWE2B0_GAMMA_NSEG       32
// Number of bits per pixel for input pixel
#define LWE2B0_GAMMA_DIBIT      10
// Number of bits per pixel for output pixel
#define LWE2B0_GAMMA_DOBIT      8
// S3.8, slope value of a gamma correction line segment
#define LWE2B0_GAMMA_SBIT       12
// S3.8, slope value of a gamma correction line segment
#define LWE2B0_GAMMA_SFBIT      8
// RGB2YUV and Color Adjuster
// This is implemented as 3x3 matrix with Y and C gains
//   Max Y and C gains of 2.0
//   Hue rotation of full 360 degree
// U0.8 Red-to-Y
#define LWE2B0_CSC_R2Y_KBIT     8
// U1.8 Green-to-Y
#define LWE2B0_CSC_G2Y_KBIT     9
// U0.8 Blue-to-Y
#define LWE2B0_CSC_B2Y_KBIT     8
// 8.0 2's complement Y Offset Gain
#define LWE2B0_CSC_YOFF_KBIT    8
// S0.8 Red-to-U
#define LWE2B0_CSC_R2U_KBIT     9
// S0.8 Green-to-U
#define LWE2B0_CSC_G2U_KBIT     9
// S0.8 Blue-to-U
#define LWE2B0_CSC_B2U_KBIT     9
// S0.8 Red-to-V
#define LWE2B0_CSC_R2V_KBIT     9
// S0.8 Green-to-V
#define LWE2B0_CSC_G2V_KBIT     9
// S0.8 Blue-to-V
#define LWE2B0_CSC_B2V_KBIT     9
// Number of bits per pixel
#define LWE2B0_CSC_DBIT 8
// Number of fractional bits of the coefficients
#define LWE2B0_CSC_FBIT 8
// Number of fractional bits of the products
#define LWE2B0_CSC_FPBIT        4
// Pixel resolution in the ISP output
// Number of bits per ISP output pixel
#define LWE2B0_OUT_DPBIT        8
// Statistics gathering
// Bit-width of aclwmulators for auto-focus measurement
#define LWE2B0_DW4ACCBIT        32
// Flicker Band Detection
// Bit width to represent max number of elements in column vector
#define LWE2B0_FB_COL_VEC_BIT   8
// Bit width of low pass coefficients
#define LWE2B0_FB_KBIT  4
// Number of fractional bits of the coefficients
#define LWE2B0_FB_FBIT  5
// some macros used by sw
// isp timing window constraints
#define LWE2B0_MIN_HBLANK_PERIOD        19
#define LWE2B0_MIN_HACTIVE_END_TO_HSCAN_END_WIDTH       9
#define LWE2B0_MIN_VACTIVE_END_TO_VSCAN_END_HEIGHT      5
// align 256;

#define LWE2B0_INCR_SYNCPT_NB_CONDS     9

#define LWE2B0_INCR_SYNCPT_0                                           (0x00000000)
// Condition mapped from raise/wait
#define LWE2B0_INCR_SYNCPT_0_COND                                      15:8
#define LWE2B0_INCR_SYNCPT_0_COND_IMMEDIATE                            (0)
#define LWE2B0_INCR_SYNCPT_0_COND_OP_DONE                              (1)
#define LWE2B0_INCR_SYNCPT_0_COND_RD_DONE                              (2)
#define LWE2B0_INCR_SYNCPT_0_COND_REG_WR_SAFE                          (3)
#define LWE2B0_INCR_SYNCPT_0_COND_FRAME_START                          (4)
#define LWE2B0_INCR_SYNCPT_0_COND_OUTPUT_END                           (5)
#define LWE2B0_INCR_SYNCPT_0_COND_FRAME_END                            (6)
#define LWE2B0_INCR_SYNCPT_0_COND_STATS                                (7)
#define LWE2B0_INCR_SYNCPT_0_COND_FB_STATS                             (8)
#define LWE2B0_INCR_SYNCPT_0_COND_COND_9                               (9)
#define LWE2B0_INCR_SYNCPT_0_COND_COND_10                              (10)
#define LWE2B0_INCR_SYNCPT_0_COND_COND_11                              (11)
#define LWE2B0_INCR_SYNCPT_0_COND_COND_12                              (12)
#define LWE2B0_INCR_SYNCPT_0_COND_COND_13                              (13)
#define LWE2B0_INCR_SYNCPT_0_COND_COND_14                              (14)
#define LWE2B0_INCR_SYNCPT_0_COND_COND_15                              (15)

// syncpt index value
#define LWE2B0_INCR_SYNCPT_0_INDX                                      7:0

#define LWE2B0_INCR_SYNCPT_CNTRL_0                                     (0x1)
// If NO_STALL is 1, then when fifos are full,
// INCR_SYNCPT methods will be dropped and the
// INCR_SYNCPT_ERROR[COND] bit will be set.
// If NO_STALL is 0, then when fifos are full,
// the client host interface will be stalled.
#define LWE2B0_INCR_SYNCPT_CNTRL_0_INCR_SYNCPT_NO_STALL                8:8

// If SOFT_RESET is set, then all internal state
// of the client syncpt block will be reset.
// To do soft reset, first set SOFT_RESET of
// all host1x clients affected, then clear all
// SOFT_RESETs.
#define LWE2B0_INCR_SYNCPT_CNTRL_0_INCR_SYNCPT_SOFT_RESET              0:0

#define LWE2B0_INCR_SYNCPT_ERROR_0                                     (0x2)
// COND_STATUS[COND] is set if the fifo for COND overflows.
// This bit is sticky and will remain set until cleared.
// Cleared by writing 1.
#define LWE2B0_INCR_SYNCPT_ERROR_0_COND_STATUS                         31:0

// just in case names were redefined using macros

#define LWE2B0_CONT_SYNCPT_FB_STATS_0                                  (0x8)
// return INDX (set HOST_CLRD packet TYPE field to SYNCPT)
#define LWE2B0_CONT_SYNCPT_FB_STATS_0_FB_STATS_INDX                    7:0

// on host read bus every time FB_STATS condition is true and FB_STATS_EN is set
#define LWE2B0_CONT_SYNCPT_FB_STATS_0_FB_STATS_EN                      8:8

// For host interface
// Context switch register.  Should be common to all modules.  Includes the
// current channel/class (which is writable by SW) and the next channel/class
// (which the hardware sets when it receives a context switch).
// Context switch works like this:
// Any context switch request triggers an interrupt to the host and causes the
// new channel/class to be stored in NEXT_CHANNEL/NEXT_CLASS (see
// vmod/chexample).  SW sees that there is a context switch interrupt and does
// the necessary operations to make the module ready to receive traffic from
// the new context.  It clears the context switch interrupt and writes
// LWRR_CHANNEL/CLASS to the same value as NEXT_CHANNEL/CLASS, which causes a
// context switch acknowledge packet to be sent to the host.  This completes
// the context switch and allows the host to continue sending data to the
// module.
// Context switches can also be pre-loaded.  If LWRR_CLASS/CHANNEL are written
// and updated to the next CLASS/CHANNEL before the context switch request
// oclwrs, an acknowledge will be generated by the module and no interrupt will
// be triggered.  This is one way for software to avoid dealing with context
// switch interrupts.
// Another way to avoid context switch interrupts is to set the AUTO_ACK bit.
// This bit tells the module to automatically acknowledge any incoming context
// switch requests without triggering an interrupt.  LWRR_* and NEXT_* will be
// updated by the module so they will always be current.

#define LWE2B0_CTXSW_0                     (0x9)
// Current working class
#define LWE2B0_CTXSW_0_LWRR_CLASS                    9:0

// Automatically acknowledge any incoming context switch requests
#define LWE2B0_CTXSW_0_AUTO_ACK                      11:11
#define LWE2B0_CTXSW_0_AUTO_ACK_MANUAL                     (0)
#define LWE2B0_CTXSW_0_AUTO_ACK_AUTOACK                    (1)

// Current working channel, reset to 'invalid'
#define LWE2B0_CTXSW_0_LWRR_CHANNEL                  15:12

// Next requested class
#define LWE2B0_CTXSW_0_NEXT_CLASS                    25:16

// Next requested channel
#define LWE2B0_CTXSW_0_NEXT_CHANNEL                  31:28

// Interrupt registers

//  This reflects status of all pending
//  interrupts which is valid as long as
//  the interrupt is not cleared even if the
//  interrupt is masked. A pending interrupt
//  can be cleared by writing a '1' to this
//  the corresponding interrupt status bit
//  in this register.
#define LWE2B0_ISP_INT_STATUS_0                    (0xa)
// Context switch interrupt status
//  (this is cleared on write)
//  0 = interrupt not pending
//  1 = interrupt pending
#define LWE2B0_ISP_INT_STATUS_0_CTXSW_INT                    0:0

// Frame End Interrupt status
// This interrupt is generated when line
//  reaches V_SCAN_SIZE value.
//  0 = interrupt not pending
//  1 = interrupt pending
#define LWE2B0_ISP_INT_STATUS_0_FRAME_END_INT                        1:1

// Vertical Blank Interrupt status
// This interrupt is generated during
//  vertical blank period.
//  0 = interrupt not pending
//  1 = interrupt pending
#define LWE2B0_ISP_INT_STATUS_0_V_BLANK_INT                  2:2

// Horizontal Blank Interrupt status
// This interrupt is generated during
//  horizontal blank period.
//  0 = interrupt not pending
//  1 = interrupt pending
#define LWE2B0_ISP_INT_STATUS_0_H_BLANK_INT                  3:3

// M2/M3/M4 Statistics Interrupt status
// This interrupt is generated at the end of
//  M2/M3/M4 statistic gathering command.
//  0 = interrupt not pending
//  1 = interrupt pending
#define LWE2B0_ISP_INT_STATUS_0_STATS_INT                    4:4

// FB Statistics Interrupt status
// This interrupt is generated at the end of
//  FB statistic gathering command.
//  0 = interrupt not pending
//  1 = interrupt pending
#define LWE2B0_ISP_INT_STATUS_0_FB_INT                       5:5

// Setting bits in this register masked the
//  corresponding interrupt but does not
//  clear a pending interrupt and does not
//  prevent a pending interrupt to be generated.
//  Masking an interrupt also does not clear
//  a pending interrupt status and does not
//  prevent interrupt status to be generated.
#define LWE2B0_ISP_INT_MASK_0                      (0xb)
// Context Switch Interrupt Mask  0 = interrupt masked
//  1 = interrupt not masked
#define LWE2B0_ISP_INT_MASK_0_CTXSW_INT_MASK                 0:0
#define LWE2B0_ISP_INT_MASK_0_CTXSW_INT_MASK_MASKED                        (0)
#define LWE2B0_ISP_INT_MASK_0_CTXSW_INT_MASK_NOTMASKED                     (1)

// Frame End Interrupt Mask  0 = interrupt masked
//  1 = interrupt not masked
#define LWE2B0_ISP_INT_MASK_0_FRAME_END_INT_MASK                     1:1
#define LWE2B0_ISP_INT_MASK_0_FRAME_END_INT_MASK_MASKED                    (0)
#define LWE2B0_ISP_INT_MASK_0_FRAME_END_INT_MASK_NOTMASKED                 (1)

// Vertical Blank Interrupt Mask  0 = interrupt masked
//  1 = interrupt not masked
#define LWE2B0_ISP_INT_MASK_0_V_BLANK_INT_MASK                       2:2
#define LWE2B0_ISP_INT_MASK_0_V_BLANK_INT_MASK_MASKED                      (0)
#define LWE2B0_ISP_INT_MASK_0_V_BLANK_INT_MASK_NOTMASKED                   (1)

// Horizontal Blank Interrupt Mask  0 = interrupt masked
//  1 = interrupt not masked
#define LWE2B0_ISP_INT_MASK_0_H_BLANK_INT_MASK                       3:3
#define LWE2B0_ISP_INT_MASK_0_H_BLANK_INT_MASK_MASKED                      (0)
#define LWE2B0_ISP_INT_MASK_0_H_BLANK_INT_MASK_NOTMASKED                   (1)

// M2/M3/M4 Statistics Interrupt Mask  0 = interrupt masked
//  1 = interrupt not masked
#define LWE2B0_ISP_INT_MASK_0_STATS_INT_MASK                 4:4
#define LWE2B0_ISP_INT_MASK_0_STATS_INT_MASK_MASKED                        (0)
#define LWE2B0_ISP_INT_MASK_0_STATS_INT_MASK_NOTMASKED                     (1)

// FB Statistics Interrupt Mask  0 = interrupt masked
//  1 = interrupt not masked
#define LWE2B0_ISP_INT_MASK_0_FB_INT_MASK                    5:5
#define LWE2B0_ISP_INT_MASK_0_FB_INT_MASK_MASKED                   (0)
#define LWE2B0_ISP_INT_MASK_0_FB_INT_MASK_NOTMASKED                        (1)

// Setting bits in this register enable
//  the corresponding interrrupt event to
//  generate a pending interrupt. Interrupt
//  output signal will be activated only if
//  the corresponding interrupt is not masked.
//  Disabling an interrupt will not clear
//  a corresponding pending interrupt - it
//  only prevent a new interrupt event to
//  generate a pending interrupt.
#define LWE2B0_ISP_INT_ENABLE_0                    (0xc)
// Context Switch Interrupt Enable  0 = interrupt disabled
//  1 = interrupt enabled
#define LWE2B0_ISP_INT_ENABLE_0_CTXSW_INT_ENABLE                     0:0
#define LWE2B0_ISP_INT_ENABLE_0_CTXSW_INT_ENABLE_DISABLE                   (0)
#define LWE2B0_ISP_INT_ENABLE_0_CTXSW_INT_ENABLE_ENABLE                    (1)

// Frame End Interrupt Enable  0 = interrupt disabled
//  1 = interrupt enabled
#define LWE2B0_ISP_INT_ENABLE_0_FRAME_END_INT_ENABLE                 1:1
#define LWE2B0_ISP_INT_ENABLE_0_FRAME_END_INT_ENABLE_DISABLE                       (0)
#define LWE2B0_ISP_INT_ENABLE_0_FRAME_END_INT_ENABLE_ENABLE                        (1)

// Vertical Blank Interrupt Enable  0 = interrupt disabled
//  1 = interrupt enabled
#define LWE2B0_ISP_INT_ENABLE_0_V_BLANK_INT_ENABLE                   2:2
#define LWE2B0_ISP_INT_ENABLE_0_V_BLANK_INT_ENABLE_DISABLE                 (0)
#define LWE2B0_ISP_INT_ENABLE_0_V_BLANK_INT_ENABLE_ENABLE                  (1)

// Horizontal Blank Interrupt Enable  0 = interrupt disabled
//  1 = interrupt enabled
#define LWE2B0_ISP_INT_ENABLE_0_H_BLANK_INT_ENABLE                   3:3
#define LWE2B0_ISP_INT_ENABLE_0_H_BLANK_INT_ENABLE_DISABLE                 (0)
#define LWE2B0_ISP_INT_ENABLE_0_H_BLANK_INT_ENABLE_ENABLE                  (1)

// M2/M3/M4 Statistics Interrupt Enable  0 = interrupt disabled
//  1 = interrupt enabled
#define LWE2B0_ISP_INT_ENABLE_0_STATS_INT_ENABLE                     4:4
#define LWE2B0_ISP_INT_ENABLE_0_STATS_INT_ENABLE_DISABLE                   (0)
#define LWE2B0_ISP_INT_ENABLE_0_STATS_INT_ENABLE_ENABLE                    (1)

// FB Statistics Interrupt Enable  0 = interrupt disabled
//  1 = interrupt enabled
#define LWE2B0_ISP_INT_ENABLE_0_FB_INT_ENABLE                        5:5
#define LWE2B0_ISP_INT_ENABLE_0_FB_INT_ENABLE_DISABLE                      (0)
#define LWE2B0_ISP_INT_ENABLE_0_FB_INT_ENABLE_ENABLE                       (1)

// Signal raise command

#define LWE2B0_ISP_SIGNAL_RAISE_0                  (0xd)
// Signal Raise select  0 = FRAME_START, return raise vector if line
//      counter is equal to 1.
//   This can be used when host needs to
//   do a long sequence of register writes 
//   (e.g. switching between preview and
//   snapshot).
//   The new configuration will take place
//   two frames later, meaning host has 
//   about one video frame time to program ISP.
//  1 = OUTPUT_END, return raise vector if line
//      counter is equal to OUTPUT_FRAME_HEIGHT.
//   This is a more aggressive raise than 
//   FRAME_END; it gives more time for host
//   to do register writes in between frames
//   if the number of register writes are
//   relatively long (e.g. color correction
//   matrix, gamma table, etc).
//   The new configuration will take place
//   at next frame.
//   OUTPUT_END should be used only for writing
//   registers that influence the pixel
//   content (such as gamma, CSC, etc), but
//   not for registers that change the timing
//   or for memory IO (like demosaic or
//   statistics gathering).
//  2 = FRAME_END, return raise vector if line
//      counter is equal to V_SCAN_SIZE.
//   This can be used if the number of 
//   register writes is small (e.g. WB gains).
//   The new configuration will take place
//   at next frame.
#define LWE2B0_ISP_SIGNAL_RAISE_0_SIGNAL_RAISE_SELECT                        1:0
#define LWE2B0_ISP_SIGNAL_RAISE_0_SIGNAL_RAISE_SELECT_FRAME_START                  (0)
#define LWE2B0_ISP_SIGNAL_RAISE_0_SIGNAL_RAISE_SELECT_OUTPUT_END                   (1)
#define LWE2B0_ISP_SIGNAL_RAISE_0_SIGNAL_RAISE_SELECT_FRAME_END                    (2)

// Channel that issues this Raise
// This channel ID is returned when the
// programmed signal raise event oclwrred.
#define LWE2B0_ISP_SIGNAL_RAISE_0_RAISE_CHANNEL_ID                   7:4

// Signal Raise Vector
// This raise vector is returned when the
// programmed signal raise event oclwrred.
#define LWE2B0_ISP_SIGNAL_RAISE_0_SIGNAL_RAISE_VECTOR                        12:8

// When written,  a REFCOUNT to be sent to the host.  
// This will be conditioned with event (e.g. frame_end), but that is not yet implemented.
// Software must not write this register if a previous request (from previous write) is
// still outstanding.
// This register is not used for AP15
//reg ISP_SIGNAL_REFCOUNT                 incr1
//    LW_REFCNT_VALUE_BITS-1:0  rw  SIGNAL_REFCOUNT                   // bit number to refcount
//;
// High priority memory control interface

//  as well as read clients
#define LWE2B0_ISP_HP_THRESHOLD_0                  (0xe)
// Write threshold value applicable to
//  write client
#define LWE2B0_ISP_HP_THRESHOLD_0_CSW_ISPW2MC_HPTH                   6:0

// Read threshold value applicable to
//  read client
#define LWE2B0_ISP_HP_THRESHOLD_0_CSR_ISPR2MC_HPTH                   15:8

// Memory High Priority timer count
#define LWE2B0_ISP_HP_THRESHOLD_0_CSR_ISPR2MC_HPTM                   21:16

// Control registers.
// Most processing blocks shall bypass what comes in and relay it to the next block.

#define LWE2B0_ISP_CONTROL1_0                      (0xf)
// Type of imager  0 = Bayer CFA type
//  1 = Stacked pixel type
//      (not supported in AP15)
#define LWE2B0_ISP_CONTROL1_0_IMAGER_TYPE                    2:0
#define LWE2B0_ISP_CONTROL1_0_IMAGER_TYPE_BAYER                    (0)
#define LWE2B0_ISP_CONTROL1_0_IMAGER_TYPE_STACKED                  (1)

// Frame Optical Black (FOB)  0 = disable acquisition
//  1 = enable acquisition
#define LWE2B0_ISP_CONTROL1_0_FOB_ACQUIRE_ENABLE                     3:3
#define LWE2B0_ISP_CONTROL1_0_FOB_ACQUIRE_ENABLE_DISABLE                   (0)
#define LWE2B0_ISP_CONTROL1_0_FOB_ACQUIRE_ENABLE_ENABLE                    (1)

// FOB filter self-reset   0 = disable (no reset is applied)
//  1 = enable (reset every frame)
#define LWE2B0_ISP_CONTROL1_0_FOB_SELFRESET_ENABLE                   4:4
#define LWE2B0_ISP_CONTROL1_0_FOB_SELFRESET_ENABLE_DISABLE                 (0)
#define LWE2B0_ISP_CONTROL1_0_FOB_SELFRESET_ENABLE_ENABLE                  (1)

// FOB filter selection  0 = disable temporal filtering
//  1 = enable temporal filtering
#define LWE2B0_ISP_CONTROL1_0_FOB_TEMPFILTER_ENABLE                  5:5
#define LWE2B0_ISP_CONTROL1_0_FOB_TEMPFILTER_ENABLE_DISABLE                        (0)
#define LWE2B0_ISP_CONTROL1_0_FOB_TEMPFILTER_ENABLE_ENABLE                 (1)

// Line Optical Black (LOB)  0 = disable acquisition
//  1 = enable acquisition
#define LWE2B0_ISP_CONTROL1_0_LOB_ACQUIRE_ENABLE                     6:6
#define LWE2B0_ISP_CONTROL1_0_LOB_ACQUIRE_ENABLE_DISABLE                   (0)
#define LWE2B0_ISP_CONTROL1_0_LOB_ACQUIRE_ENABLE_ENABLE                    (1)

// Fixed column noise to acquire  0 = disable acquisition
//  1 = enable acquisition
//      (not supported in AP15)
#define LWE2B0_ISP_CONTROL1_0_FCN_ACQUIRE_ENABLE                     7:7
#define LWE2B0_ISP_CONTROL1_0_FCN_ACQUIRE_ENABLE_DISABLE                   (0)
#define LWE2B0_ISP_CONTROL1_0_FCN_ACQUIRE_ENABLE_ENABLE                    (1)

// Fixed column noise compensation This is effective only when OB_FCN_ENABLE is
// enabled.
//  0 = disable
//  1 = enable
//      (not supported in AP15)
#define LWE2B0_ISP_CONTROL1_0_FCN_COMPENSATION_ENABLE                        8:8
#define LWE2B0_ISP_CONTROL1_0_FCN_COMPENSATION_ENABLE_DISABLE                      (0)
#define LWE2B0_ISP_CONTROL1_0_FCN_COMPENSATION_ENABLE_ENABLE                       (1)

// Switch for OB & FCN processing      0 = disable
//  1 = enable
#define LWE2B0_ISP_CONTROL1_0_OB_FCN_ENABLE                  9:9
#define LWE2B0_ISP_CONTROL1_0_OB_FCN_ENABLE_DISABLE                        (0)
#define LWE2B0_ISP_CONTROL1_0_OB_FCN_ENABLE_ENABLE                 (1)

// Common gain control             0 = disable
//  1 = enable
#define LWE2B0_ISP_CONTROL1_0_CMNGAIN_ENABLE                 10:10
#define LWE2B0_ISP_CONTROL1_0_CMNGAIN_ENABLE_DISABLE                       (0)
#define LWE2B0_ISP_CONTROL1_0_CMNGAIN_ENABLE_ENABLE                        (1)

// Switch for input linearization processing  0 = disable
//  1 = enable
#define LWE2B0_ISP_CONTROL1_0_DE_KNEE_ENABLE                 11:11
#define LWE2B0_ISP_CONTROL1_0_DE_KNEE_ENABLE_DISABLE                       (0)
#define LWE2B0_ISP_CONTROL1_0_DE_KNEE_ENABLE_ENABLE                        (1)

// Switch for lens shading compensation  0 = disable
//  1 = enable
#define LWE2B0_ISP_CONTROL1_0_LENS_SHADING_ENABLE                    12:12
#define LWE2B0_ISP_CONTROL1_0_LENS_SHADING_ENABLE_DISABLE                  (0)
#define LWE2B0_ISP_CONTROL1_0_LENS_SHADING_ENABLE_ENABLE                   (1)

// Switch for white balance processing  0 = disable
//  1 = enable
#define LWE2B0_ISP_CONTROL1_0_WHITE_BALANCE_ENABLE                   13:13
#define LWE2B0_ISP_CONTROL1_0_WHITE_BALANCE_ENABLE_DISABLE                 (0)
#define LWE2B0_ISP_CONTROL1_0_WHITE_BALANCE_ENABLE_ENABLE                  (1)

// Switch for operation mode  0 = dynamic (real-time) mode
//  1 = pre-programmed mode by host
#define LWE2B0_ISP_CONTROL1_0_WHITE_BALANCE_MODE                     14:14
#define LWE2B0_ISP_CONTROL1_0_WHITE_BALANCE_MODE_DYNAMIC                   (0)
#define LWE2B0_ISP_CONTROL1_0_WHITE_BALANCE_MODE_PROGRAMMED                        (1)

// Auto White Balance gains hold
//  0 = recallwlate gain factors
//  1 = hold current gain factors
#define LWE2B0_ISP_CONTROL1_0_WBHOLD                 15:15

// Top-clipping enable  0 = disable
//  1 = enable
#define LWE2B0_ISP_CONTROL1_0_TOP_CLIP_ENABLE                        16:16
#define LWE2B0_ISP_CONTROL1_0_TOP_CLIP_ENABLE_DISABLE                      (0)
#define LWE2B0_ISP_CONTROL1_0_TOP_CLIP_ENABLE_ENABLE                       (1)

// Bad pixel detection and concealment  0 = disable
//  1 = enable
#define LWE2B0_ISP_CONTROL1_0_BAD_PIXEL_CONCEAL                      17:17
#define LWE2B0_ISP_CONTROL1_0_BAD_PIXEL_CONCEAL_DISABLE                    (0)
#define LWE2B0_ISP_CONTROL1_0_BAD_PIXEL_CONCEAL_ENABLE                     (1)

// Noise Reduction This is effective only if D3X3BPNR, or
//  D5X5NR, or D5X5BPNR demosaic mode is
//  selected
//  0 = disable
//  1 = enable 
#define LWE2B0_ISP_CONTROL1_0_NOISE_REDUCTION_ENABLE                 18:18
#define LWE2B0_ISP_CONTROL1_0_NOISE_REDUCTION_ENABLE_DISABLE                       (0)
#define LWE2B0_ISP_CONTROL1_0_NOISE_REDUCTION_ENABLE_ENABLE                        (1)

// Selection of output image format   0 = demosaiced YUV or RGB image
//  1 = processed Bayer image
#define LWE2B0_ISP_CONTROL1_0_OUT_IMAGE_TYPE                 31:31
#define LWE2B0_ISP_CONTROL1_0_OUT_IMAGE_TYPE_DEMOSAICED                    (0)
#define LWE2B0_ISP_CONTROL1_0_OUT_IMAGE_TYPE_BAYER_PROCED                  (1)

// There are few demosaicing schemes built in and selectable from DEMOSAIC_MODE bits. Depending
//  on the demosaic mode selection, the number of line-delay buffers are set up.
// Edge enhancement and noise reduction are available only when the DEMOSAIC_MODE selected
//  has implied datapath.
// Notice that "DEMOSAIC_ENABLE" means "pixel-processing-enable" and not narrow sense of 
//   enable for the demosaic (color-interpolation) section.

#define LWE2B0_ISP_CONTROL2_0                      (0x10)
// Demosaicing processing  0 = disable
//  1 = enable
#define LWE2B0_ISP_CONTROL2_0_DEMOSAIC_ENABLE                        0:0
#define LWE2B0_ISP_CONTROL2_0_DEMOSAIC_ENABLE_DISABLE                      (0)
#define LWE2B0_ISP_CONTROL2_0_DEMOSAIC_ENABLE_ENABLE                       (1)

// Demosaic formulas
//  0 = Built-in fixed formulas
//  1 = Host-programmed formulas    
#define LWE2B0_ISP_CONTROL2_0_PROG_DEMOSAIC_ENABLE                   1:1

// Number of bits/pixel stored in demosaicing
//  line buffer in memory  0 = 10-bit/pixel, 12-pixel/128-bit
#define LWE2B0_ISP_CONTROL2_0_LB_DATA_WIDTH                  5:2
#define LWE2B0_ISP_CONTROL2_0_LB_DATA_WIDTH_D10                    (0)

// Demosaicing method  0 = 3X3 demosaicing
//      2-line buffer (1 write, 2 read)
//  1 = 3X3 demosaicing, bad-pixel correction,
//      and noise-reduction
//      4-line buffer (2 write, 4 read)
//  2 = 5X5 demosaicing
//      4-line buffer (2 write, 4 read)
//  3 = 5X5 demosaicing and noise-reduction
//      4-line buffer (2 write, 4 read)
//  4 = 5x5 demosaicing, bad-pixel correction,
//      and noise reduction
//      6-line buffer (2 write, 6 read)
#define LWE2B0_ISP_CONTROL2_0_DEMOSAIC_MODE                  8:6
#define LWE2B0_ISP_CONTROL2_0_DEMOSAIC_MODE_D3X3                   (0)
#define LWE2B0_ISP_CONTROL2_0_DEMOSAIC_MODE_D3X3BPNR                       (1)
#define LWE2B0_ISP_CONTROL2_0_DEMOSAIC_MODE_D5X5                   (2)
#define LWE2B0_ISP_CONTROL2_0_DEMOSAIC_MODE_D5X5NR                 (3)
#define LWE2B0_ISP_CONTROL2_0_DEMOSAIC_MODE_D5X5BPNR                       (4)

// CAR processing enable  0 = disable
//  1 = enable
#define LWE2B0_ISP_CONTROL2_0_CAR_ENABLE                     9:9
#define LWE2B0_ISP_CONTROL2_0_CAR_ENABLE_DISABLE                   (0)
#define LWE2B0_ISP_CONTROL2_0_CAR_ENABLE_ENABLE                    (1)

// Edge Enhancement  0 = disable
//  1 = enable
#define LWE2B0_ISP_CONTROL2_0_EDGE_ENHANCE_ENABLE                    10:10
#define LWE2B0_ISP_CONTROL2_0_EDGE_ENHANCE_ENABLE_DISABLE                  (0)
#define LWE2B0_ISP_CONTROL2_0_EDGE_ENHANCE_ENABLE_ENABLE                   (1)

// Color Correction  0 = disable
//  1 = enable
#define LWE2B0_ISP_CONTROL2_0_COLOR_CORRECTION_ENABLE                        11:11
#define LWE2B0_ISP_CONTROL2_0_COLOR_CORRECTION_ENABLE_DISABLE                      (0)
#define LWE2B0_ISP_CONTROL2_0_COLOR_CORRECTION_ENABLE_ENABLE                       (1)

// Noise reduction in the color correction  0 = disable
//  1 = enable
#define LWE2B0_ISP_CONTROL2_0_CC_NR_ENABLE                   12:12
#define LWE2B0_ISP_CONTROL2_0_CC_NR_ENABLE_DISABLE                 (0)
#define LWE2B0_ISP_CONTROL2_0_CC_NR_ENABLE_ENABLE                  (1)

// Negative <-> positive reversal effect  0 = disable
//  1 = enable
#define LWE2B0_ISP_CONTROL2_0_NEGATIVE_EFFECT_ENABLE                 13:13
#define LWE2B0_ISP_CONTROL2_0_NEGATIVE_EFFECT_ENABLE_DISABLE                       (0)
#define LWE2B0_ISP_CONTROL2_0_NEGATIVE_EFFECT_ENABLE_ENABLE                        (1)

// Gamma Correction  0 = disable
//  1 = enable
#define LWE2B0_ISP_CONTROL2_0_GAMMA_CORRECTION_ENABLE                        14:14
#define LWE2B0_ISP_CONTROL2_0_GAMMA_CORRECTION_ENABLE_DISABLE                      (0)
#define LWE2B0_ISP_CONTROL2_0_GAMMA_CORRECTION_ENABLE_ENABLE                       (1)

// RGB to YUV colwersion  0 = disable
//  1 = enable
#define LWE2B0_ISP_CONTROL2_0_RGB2YUV_ENABLE                 15:15
#define LWE2B0_ISP_CONTROL2_0_RGB2YUV_ENABLE_DISABLE                       (0)
#define LWE2B0_ISP_CONTROL2_0_RGB2YUV_ENABLE_ENABLE                        (1)

// ISP_ENABLE is command to capture image frames
//   0 = disable/stop capturing frame)  
//     If this is issued in the middle of capturing a frame, the frame capture
//     is terminated at the end of the frame.
//     This does not reset other register setting. Both line counter and frame
//     counter are set to 0 when frame capture is stopped.
//   1 = enable capturing of frames
//     This should be issued when frame capture is stopped. When this command is issued,
//     ISP will wait till the next V sync active edge and start capturing frame
//     from that point. Once this is issued, minimum 1 frame will be captured even if
//     this "enable" command is immediately followed by "disable" command.
// ISP_UPDATE_REQ is command to update contril registers
//   When host finishes programming of shadow registers, write 1 to this bit to indicate 
//   a new configuration data is ready for ISP to use.
//   While this bit is 1 and at the fiirst arrivel of start-of-a-frame, ISP updates its active 
//   registers with the content of shadow registers, and ISP also resets this bit to indicate
//   that the shadow registers are now free for the host for a new programming.

#define LWE2B0_ISP_COMMAND_0                       (0x11)
// ISP capture frame enable command  0 = disable/stop capturing frame)
//  1 = enable capturing of frames
#define LWE2B0_ISP_COMMAND_0_ISP_ENABLE                      0:0
#define LWE2B0_ISP_COMMAND_0_ISP_ENABLE_DISABLE                    (0)
#define LWE2B0_ISP_COMMAND_0_ISP_ENABLE_ENABLE                     (1)

// Active registers update requect command  0= no req pending/req completed
//  1= update requested/pending
#define LWE2B0_ISP_COMMAND_0_ISP_UPDATE_REQ                  31:31
#define LWE2B0_ISP_COMMAND_0_ISP_UPDATE_REQ_DISABLE                        (0)
#define LWE2B0_ISP_COMMAND_0_ISP_UPDATE_REQ_ENABLE                 (1)

// Timing Generator Control
// Whole timing frame consists of scanned part and idle parts (void periods in between lines
//  and frames).
// Overall scanned image frame is specified by H_SCAN_SIZE and V_SCAN_SIZE which includes
//  H and V blank periods. Line and pixel counters are set to 1 at transitions of active edge
//  of V and H sync input pulses.
// Note that coordinate in the whole frame starts with (1,1) instead of (0,0).  0 indicates
//  void area.
//
// In horizotal direction, H sync active edge indicates start of scan line and it initiates
//  pixel count from 1 to H_SCAN_SIZE (end of line). After H_SCAN_SIZE pixels are received,
//  the ISP module goes into void period, in which it waits for next arrival of H sync active
//  edge. In practice, H blank period is composed of inactive period during H_SCAN SIZE and the
//  void period in between end of line and start next line.
// Similarly in vertical direction, the non-blanking part consists of "optical black" lines 
//  at top and bottom of the
//  image frame and left and right of image line, and active lines with active pixels in the
//  middle.
//

#define LWE2B0_ISP_SCAN_FRAME_0                    (0x12)
// Horizontal Scan Size
//  This specifies the number of pixel clocks
//  per scan line (including H blank)
#define LWE2B0_ISP_SCAN_FRAME_0_H_SCAN_SIZE                  13:0

// Vertical Scan Size
//  This specifies the number of line periods 
//  per scan frame (including V blank)
#define LWE2B0_ISP_SCAN_FRAME_0_V_SCAN_SIZE                  29:16

// CFASEL values
// Pixel data in Bayer format consists of four color filtered pixel values {Gr, R, B, Gb} in a
// time multiplexed manner in pixel-by-pixel and line-by-line alternation.
// When an imager is given, there are four possibilities in the colors that the first active pixel 
// on the first active line starts.
//
// CFASEL_DM field is used to specify "which color to start with" in the pixel processing pipe.
// The formula to get the proper color produced is: 
//
//     CFSEL_DM = CF_Act ^ HW_CF_Modifier, 
//
// where CF_Act is the color phase of the first active pixel. 
// HW_CF_Modifier is hardware design specific value and "2'b01" for AP15 ISP. 

#define LWE2B0_ISP_TG_CONTROL_0                    (0x13)
// H sync active edge selection  0 = use positive transition (rising edge)
//  1 = use negative transition (falling edge)
#define LWE2B0_ISP_TG_CONTROL_0_HSYNC_EDGE                   0:0
#define LWE2B0_ISP_TG_CONTROL_0_HSYNC_EDGE_POS                     (0)
#define LWE2B0_ISP_TG_CONTROL_0_HSYNC_EDGE_NEG                     (1)

// V sync Active edge selection  0 = use positive transition (rising edge)
//  1 = use negative transition (falling edge)
#define LWE2B0_ISP_TG_CONTROL_0_VSYNC_EDGE                   1:1
#define LWE2B0_ISP_TG_CONTROL_0_VSYNC_EDGE_POS                     (0)
#define LWE2B0_ISP_TG_CONTROL_0_VSYNC_EDGE_NEG                     (1)

// Bayer color filter array interpretation for
//  Demosaic processing
//  The setting of this value is dependent on
//  internal hardware pipeline.  0 = R/G followed by G/B
//  1 = G/R followed by B/G
//  2 = G/B followed by R/G
//  3 = B/G followed by G/R
#define LWE2B0_ISP_TG_CONTROL_0_CFASEL_DM                    9:8
#define LWE2B0_ISP_TG_CONTROL_0_CFASEL_DM_RGGB                     (0)
#define LWE2B0_ISP_TG_CONTROL_0_CFASEL_DM_GRBG                     (1)
#define LWE2B0_ISP_TG_CONTROL_0_CFASEL_DM_GBRG                     (2)
#define LWE2B0_ISP_TG_CONTROL_0_CFASEL_DM_BGGR                     (3)


#define LWE2B0_ISP_H_ACTIVE_0                      (0x14)
// Pixel number where active line starts
#define LWE2B0_ISP_H_ACTIVE_0_ACTIVE_LINE_START                      13:0

// Number of active pixels in a line
#define LWE2B0_ISP_H_ACTIVE_0_ACTIVE_LINE_WIDTH                      29:16


#define LWE2B0_ISP_V_ACTIVE_0                      (0x15)
// Line number where active frame starts
#define LWE2B0_ISP_V_ACTIVE_0_ACTIVE_FRAME_START                     13:0

// Number of active image lines in a frame
#define LWE2B0_ISP_V_ACTIVE_0_ACTIVE_FRAME_HEIGHT                    29:16


#define LWE2B0_ISP_H_OUTPUT_0                      (0x16)
// Pixel number where output window starts
#define LWE2B0_ISP_H_OUTPUT_0_OUTPUT_LINE_START                      13:0

// Number of output window pixels in a line
#define LWE2B0_ISP_H_OUTPUT_0_OUTPUT_LINE_WIDTH                      29:16


#define LWE2B0_ISP_V_OUTPUT_0                      (0x17)
// Line number where output window starts
#define LWE2B0_ISP_V_OUTPUT_0_OUTPUT_FRAME_START                     13:0

// Number of output window lines in a frame
#define LWE2B0_ISP_V_OUTPUT_0_OUTPUT_FRAME_HEIGHT                    29:16

// Stall control
// Proper operation of ISP requires some minimum H and V blanking time.
// During the H and V blanking periods, some house keeping jobs and preparation for the next line 
// and frame processing are performed.
// Active pixel data pushed into the pixel data-pipe shall be further advanced during the H and V 
// blanking periods so that the data-pipe can freshly start next lines and frames.
// H blanking is also used for pre-fetching of pixels from line-delay memories allocated in the
// frame buffer memory.
//
// It is assumed that pixel clock is active not only during the h-scan-size period but also some time 
//   after that, so to allow control action during H-blanking period (in a vague sense).
// It is also assumed that hsync pulses are active at least during v-scan-size period.
//   Local-hsync may be generated and used for control action during V-blanking period. 
//
// As a rule of thumb, current ISP requires,
//   about 20T from end-of-current-active-line to the start-of-next-active-line,
//   about 10T from end-of-current-active-line to the end-of-current-h-scan-size,
//   2H to 5H (depending on the demosdaic mode) from end-of-current-active-frame to the next v-sync 
//   (Start-of-Frame).
//   where T and H represent pixel-clock and line periods.
//
// The DVS bus, vi2isp, that supplies input data to ISP can be stalled.
// This stall control register specifies how ISP generates the stall signal to the vi2isp DVS bus.
//
//   HBLANK_ACT_MIN[7:0] is to specify minimum H-blanking width (pixels) from end-of-current-active-line 
//     to the next h-sync (Start-of-Line) in 0 to 255 pixels.
//   HBLANK_SCAN_MIN[7:0] is to specify minimum H-blanking width (pixels) from end-of-h-scan-size to 
//     the next h-sync (Start-of-Line)
//   VBLANK_ACT_MIN[3:0] is to specify minimum V-blanking width (lines) from end-of-current-active-frame 
//     to the next v-sync (Start-of-Frame) in 0 to 15 lines.
//   VBLANK_SCAN_MIN[3:0] is to specify minimum V-blanking width (lines) from end-of-v-scan-size to the 
//     next v-sync (Start-of-Frame) in 0 to 15 lines.
//
//   LOCAL_HSYNC_ENABLE = 1 activates local substitution of hsync during V-Blanking 
//   MEMORY_STALL_ACTIVE =1 activates stall when data from frame buffer is not ready when needed.
//     (AP15 does not use this bit since it has dedicated local SRAM.)
//   HBLANK_ACT_STALL_ACTIVE = 1 activates HBLANK_ACT_MIN condition.
//   HBLANK_SCAN_STALL_ACTIVE = 1 activates HBLANK_SCAN_MIN condition.
//   VBLANK_ACT_STALL_ACTIVE = 1 activates VBLANK_ACT_MIN condition.
//   VBLANK_SCAN_STALL_ACTIVE = 1 activates VBLANK_SCAN_MIN condition.
//
// When a condition is not satisfied and the stall_active bit is asserted, the stall is created.
//
// Be reminded of the following facts.
//  * HBLANK_ACT_STALL_ACTIVE and HBLANK_SCAN_STALL_ACTIVE are independent for control of H-lanking.
//  * VBLANK_ACT_STALL_ACTIVE and VBLANK_SCAN_STALL_ACTIVE are independent for control of V-lanking.
//  * When LOCAL_HSYNS_ENABLE = 1, hsync period locally generated is equal to the wider one dictated 
//      by HBLANK_ACT_MIN and HBLANK_SCAN_MIN.
//  * When LOCAL_HSYNC_ENABLE = 1, stall is created in all H-blanking if input (active) hsync period 
//      is shorter that that of locally generated hsync pulses. 
//    (LOCAL_HSYNC_ENABLE = 1 takes precidence to HBLAN_ACT_STALL_ACTIVE and HBLAN_SCAN_STALL_ACTIVE.) 

#define LWE2B0_ISP_STALL_CONTROL_0                 (0x18)
// Active-line-end to SL
#define LWE2B0_ISP_STALL_CONTROL_0_HBLANK_ACT_MIN                    7:0

// h_scan_size_end to SL
#define LWE2B0_ISP_STALL_CONTROL_0_HBLANK_SCAN_MIN                   15:8

// Active-frame-end to SF
#define LWE2B0_ISP_STALL_CONTROL_0_VBLANK_ACT_MIN                    19:16

// v_scan_size_end to SF
#define LWE2B0_ISP_STALL_CONTROL_0_VBLANK_SCAN_MIN                   23:20

//  0 = disable
//  1 = enable
#define LWE2B0_ISP_STALL_CONTROL_0_LOCAL_HSYNC_ENABLE                        24:24
#define LWE2B0_ISP_STALL_CONTROL_0_LOCAL_HSYNC_ENABLE_DISABLE                      (0)
#define LWE2B0_ISP_STALL_CONTROL_0_LOCAL_HSYNC_ENABLE_ENABLE                       (1)

//  0 = disable
//  1 = enable
//  (MEMORY_STALL is not used in AP15)
#define LWE2B0_ISP_STALL_CONTROL_0_MEMORY_STALL_ACTIVE                       27:27
#define LWE2B0_ISP_STALL_CONTROL_0_MEMORY_STALL_ACTIVE_DISABLE                     (0)
#define LWE2B0_ISP_STALL_CONTROL_0_MEMORY_STALL_ACTIVE_ENABLE                      (1)

//   0 = disable
//  1 = enable
#define LWE2B0_ISP_STALL_CONTROL_0_HBLANK_ACT_STALL_ACTIVE                   28:28
#define LWE2B0_ISP_STALL_CONTROL_0_HBLANK_ACT_STALL_ACTIVE_DISABLE                 (0)
#define LWE2B0_ISP_STALL_CONTROL_0_HBLANK_ACT_STALL_ACTIVE_ENABLE                  (1)

//  0 = disable
//  1 = enable
#define LWE2B0_ISP_STALL_CONTROL_0_HBLANK_SCAN_STALL_ACTIVE                  29:29
#define LWE2B0_ISP_STALL_CONTROL_0_HBLANK_SCAN_STALL_ACTIVE_DISABLE                        (0)
#define LWE2B0_ISP_STALL_CONTROL_0_HBLANK_SCAN_STALL_ACTIVE_ENABLE                 (1)

//  0 = disable
//  1 = enable
#define LWE2B0_ISP_STALL_CONTROL_0_VBLANK_ACT_STALL_ACTIVE                   30:30
#define LWE2B0_ISP_STALL_CONTROL_0_VBLANK_ACT_STALL_ACTIVE_DISABLE                 (0)
#define LWE2B0_ISP_STALL_CONTROL_0_VBLANK_ACT_STALL_ACTIVE_ENABLE                  (1)

//  0 = disable
//  1 = enable
#define LWE2B0_ISP_STALL_CONTROL_0_VBLANK_SCAN_STALL_ACTIVE                  31:31
#define LWE2B0_ISP_STALL_CONTROL_0_VBLANK_SCAN_STALL_ACTIVE_DISABLE                        (0)
#define LWE2B0_ISP_STALL_CONTROL_0_VBLANK_SCAN_STALL_ACTIVE_ENABLE                 (1)

// Optical Black and Fixed Column Noise 
// Two types of optical black: 
//  Frame OB (FOB) is acquired by averaging pixel values at OB_TOP region.  
//  If FOB is enabled (in ISP_CONTROL1), all pixels in active area will be subtraced by FOB.
//  Line OB (LOB) is acquired by averaging pixel values at OB_LEFT region.
//  If LOB is enabled (in ISP_CONTROL1), for each scanline an LOB value will first be acquired,
//  and the following pixels in this scanline will be subtracted by this LOB.  
//  In FANDL mode, both FOB and LOB will be acquired, and the final OB value for each pixel
//  are (FOB+LOB)/2.
//  In AP15, the following features are not supported:
//  (1) OB_RIGHT
//  (2) FCN
// Optical black restoration
// (Manual) adjust value is individualized for {r, gr, gb, b} colors
//    Adjust-values are pos/neg 2's complement in (ISP_OBR_DBIT+1) bits
// Effective_OB_value that is subtracted from the raw pixel value is,
//    Effective_OB_value = (measured_OB_value) + (manual_adjust_value)
// I.e. positive adjust_value increases effective_OB_level.
// 
// In AP15, 4-bit FOB_FLTR_COEF and TOB_FLTR_COEF are valid in {0,1,2,3,4,5,6,7}.
//   Out of range values in [8,15] are treated as value 0. 

// processing
#define LWE2B0_ISP_OB_FCN_CONTROL1_0                       (0x19)
// U0.W, Filter coefficient of FOB acquisition.
#define LWE2B0_ISP_OB_FCN_CONTROL1_0_FOB_FLTR_COEF                   3:0

// U0.W, Filter coefficient of FOB temporal
//  filter.
#define LWE2B0_ISP_OB_FCN_CONTROL1_0_TOB_FLTR_COEF                   7:4

// U0.W, Filter coefficient of FCN acquisition.
#define LWE2B0_ISP_OB_FCN_CONTROL1_0_FCN_FLTR_COEF                   11:8

// U0.W, Filter coefficient of column filter.
//    (ISP_OB_DBIT+23):24 rw  OB_ADJUST           // Manual adjustment applied to the black level 
//                                                //  (positive only)
//                                                //  (FCN is not supported in AP15)
#define LWE2B0_ISP_OB_FCN_CONTROL1_0_CLMN_FLTR_COEF                  19:16


// Manual_adjust_values 
#define LWE2B0_ISP_OB_FCN_CONTROL2_0                       (0x1a)
// For R pixels           
#define LWE2B0_ISP_OB_FCN_CONTROL2_0_OB_R1_ADJUST                    10:0

// For B pixels           
//    31  rw  INDIVIDUAL_OB_ADJUST        init=0  // 1 to enable individualized OB adjust
//                                                // 0 to enable common OB adjust
#define LWE2B0_ISP_OB_FCN_CONTROL2_0_OB_B1_ADJUST                    26:16


// Manual_adjust_values 
#define LWE2B0_ISP_OB_FCN_CONTROL3_0                       (0x1b)
// For G pixels on R/G rows           
#define LWE2B0_ISP_OB_FCN_CONTROL3_0_OB_G1_ADJUST                    10:0

// For G pixels on B/G rows           
#define LWE2B0_ISP_OB_FCN_CONTROL3_0_OB_G2_ADJUST                    26:16


// Manual_adjust_values 
#define LWE2B0_ISP_OB_FCN_CONTROL4_0                       (0x1c)
// LOB, number of columns  0 = 2 columns
//  1 = 4 columns
//  2 = 8 columns
#define LWE2B0_ISP_OB_FCN_CONTROL4_0_LOB_WIDTH                       1:0
#define LWE2B0_ISP_OB_FCN_CONTROL4_0_LOB_WIDTH_LOB2                        (0)
#define LWE2B0_ISP_OB_FCN_CONTROL4_0_LOB_WIDTH_LOB4                        (1)
#define LWE2B0_ISP_OB_FCN_CONTROL4_0_LOB_WIDTH_LOB8                        (2)

// Optical Black (OB) extraction mode  0 = Frame-OB
//  1 = Frame_OB and Line-OB mixed
//  2 = Line-OB
#define LWE2B0_ISP_OB_FCN_CONTROL4_0_OB_MODE                 4:2
#define LWE2B0_ISP_OB_FCN_CONTROL4_0_OB_MODE_FRAME                 (0)
#define LWE2B0_ISP_OB_FCN_CONTROL4_0_OB_MODE_FANDL                 (1)
#define LWE2B0_ISP_OB_FCN_CONTROL4_0_OB_MODE_LINE                  (2)


#define LWE2B0_ISP_OB_TOP_0                        (0x1d)
// Line number where top optical black area
//  starts. Unreachable large value will 
// effectively disable top OB.
#define LWE2B0_ISP_OB_TOP_0_OB_TOP_START                     13:0

// Number of top optical black lines
//  valid values are [1..15]. 0 means no OB values
//  are acquired at top
#define LWE2B0_ISP_OB_TOP_0_OB_TOP_HEIGHT                    19:16


#define LWE2B0_ISP_OB_BOTTOM_0                     (0x1e)
// Line number where bottom optical black area
//  starts. Unreachable large value will  
// effectively disable bottom OB. 
#define LWE2B0_ISP_OB_BOTTOM_0_OB_BOTTOM_START                       13:0

// Number of bottom optical black lines
#define LWE2B0_ISP_OB_BOTTOM_0_OB_BOTTOM_HEIGHT                      19:16


#define LWE2B0_ISP_OB_LEFT_0                       (0x1f)
// Pixel number where left optical black area
//  starts. Unreachable large value will 
// effectively disable left OB.
#define LWE2B0_ISP_OB_LEFT_0_OB_LEFT_START                   13:0

// Number of left optical black pixels
//  valid values are 2, 4, or 8, and it has to
//  be consistent with LOB_WIDTH in ISP_CONTROL1
#define LWE2B0_ISP_OB_LEFT_0_OB_LEFT_WIDTH                   19:16


#define LWE2B0_ISP_OB_RIGHT_0                      (0x20)
// Pixel number where right optical black area
//  starts
#define LWE2B0_ISP_OB_RIGHT_0_OB_RIGHT_START                 13:0

// Number of right optical black pixels
//  (RIGHT OB are not used in AP15)
#define LWE2B0_ISP_OB_RIGHT_0_OB_RIGHT_WIDTH                 19:16

// Common gain control
// This single gain value is intended to be a part of exposure control and applied 
//   commonly to pixels of all colors.
// Total length is ISP_CG_DBIT bits and the lower ISP_CG_FBIT bits are fraction.
//   I.e.  U(ISP_CG_DBIT-ISP_CG_FBIT).(ISP_CG_FBIT)

#define LWE2B0_ISP_COMMON_GAIN_CONTROL_0                   (0x21)
// U(ISP_CG_DBIT-ISP_CG_FBIT).(ISP_CG_FBIT)
#define LWE2B0_ISP_COMMON_GAIN_CONTROL_0_CMNGAIN                     9:0

// De-Knee, input linearization
// This is implemented as piece-wise linear approximation of up to 16 segments.
// Line definitions must be provided in ascending order (of input range) from CONFIG1A to
// CONFIG16B.
//
// Deknee characteristics can be specified flexibly in arbitrary number of line segments
// Number of line-segments actively used may be arbitrary (from 1 to 16).
// Each line-segment may cover arbitrarily wide input range. Hence, each line-segment is
//   specified by (input-start-value, output-start-value, slope-value).
// Input-start-values for the 16 line-segments are specified by DEKNEE_IX,
//   where X is in {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}.
//   DEKNEE_I1 is always ZERO, so it need not to be explicitly specified.
// Output-start-values are output values correspond to the input-start-value.
// X-th line-segment has the input-range [DEKNEE_IX, DEKNEE_I(X+1)), except for X=16.
//   16-th line-segment has the input range [DEKNEE_I16, 2^ISP_PIXEL_DIBIT-1].
//
// Note that DEKNEE_IX must be given in an ascending order.
// For those line-segments not used, "input-start-value" and "output-start-value" must be 
//   specified as 2^ISP_PIXEL_DIBIT-1 (maximum input value) and the corresponding output value. 
//
// There are 4 sets of de-knee configuration registers for {Gr, Gb, R, B} color groups.   

//    (ISP_PIXEL_DIBIT-1):0   rw  DEKNEE_GR_I1    // Input start value of #1 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR1A_0                    (0x22)
// Output start value of #1 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR1A_0_DEKNEE_GR_O1                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GR1B_0                    (0x23)
// Slope value of #1 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR1B_0_DEKNEE_GR_S1                 11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_GR2A_0                    (0x24)
// Input start value of #2 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR2A_0_DEKNEE_GR_I2                 9:0

// Output start value of #2 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR2A_0_DEKNEE_GR_O2                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GR2B_0                    (0x25)
// Slope value of #2 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR2B_0_DEKNEE_GR_S2                 11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_GR3A_0                    (0x26)
// Input start value of #3 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR3A_0_DEKNEE_GR_I3                 9:0

// Output start value of #3 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR3A_0_DEKNEE_GR_O3                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GR3B_0                    (0x27)
// Slope value of #3 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR3B_0_DEKNEE_GR_S3                 11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_GR4A_0                    (0x28)
// Input start value of #4 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR4A_0_DEKNEE_GR_I4                 9:0

// Output start value of #4 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR4A_0_DEKNEE_GR_O4                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GR4B_0                    (0x29)
// Slope value of #4 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR4B_0_DEKNEE_GR_S4                 11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_GR5A_0                    (0x2a)
// Input start value of #5 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR5A_0_DEKNEE_GR_I5                 9:0

// Output start value of #5 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR5A_0_DEKNEE_GR_O5                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GR5B_0                    (0x2b)
// Slope value of #5 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR5B_0_DEKNEE_GR_S5                 11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_GR6A_0                    (0x2c)
// Input start value of #6 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR6A_0_DEKNEE_GR_I6                 9:0

// Output start value of #6 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR6A_0_DEKNEE_GR_O6                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GR6B_0                    (0x2d)
// Slope value of #6 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR6B_0_DEKNEE_GR_S6                 11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_GR7A_0                    (0x2e)
// Input start value of #7 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR7A_0_DEKNEE_GR_I7                 9:0

// Output start value of #7 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR7A_0_DEKNEE_GR_O7                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GR7B_0                    (0x2f)
// Slope value of #7 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR7B_0_DEKNEE_GR_S7                 11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_GR8A_0                    (0x30)
// Input start value of #8 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR8A_0_DEKNEE_GR_I8                 9:0

// Output start value of #8 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR8A_0_DEKNEE_GR_O8                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GR8B_0                    (0x31)
// Slope value of #8 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR8B_0_DEKNEE_GR_S8                 11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_GR9A_0                    (0x32)
// Input start value of #9 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR9A_0_DEKNEE_GR_I9                 9:0

// Output start value of #9 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR9A_0_DEKNEE_GR_O9                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GR9B_0                    (0x33)
// Slope value of #9 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR9B_0_DEKNEE_GR_S9                 11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_GR10A_0                   (0x34)
// Input start value of #10 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR10A_0_DEKNEE_GR_I10                       9:0

// Output start value of #10 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR10A_0_DEKNEE_GR_O10                       25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GR10B_0                   (0x35)
// Slope value of #10 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR10B_0_DEKNEE_GR_S10                       11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_GR11A_0                   (0x36)
// Input start value of #11 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR11A_0_DEKNEE_GR_I11                       9:0

// Output start value of #11 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR11A_0_DEKNEE_GR_O11                       25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GR11B_0                   (0x37)
// Slope value of #11 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR11B_0_DEKNEE_GR_S11                       11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_GR12A_0                   (0x38)
// Input start value of #12 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR12A_0_DEKNEE_GR_I12                       9:0

// Output start value of #12 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR12A_0_DEKNEE_GR_O12                       25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GR12B_0                   (0x39)
// Slope value of #12 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR12B_0_DEKNEE_GR_S12                       11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_GR13A_0                   (0x3a)
// Input start value of #13 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR13A_0_DEKNEE_GR_I13                       9:0

// Output start value of #13 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR13A_0_DEKNEE_GR_O13                       25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GR13B_0                   (0x3b)
// Slope value of #13 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR13B_0_DEKNEE_GR_S13                       11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_GR14A_0                   (0x3c)
// Input start value of #14 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR14A_0_DEKNEE_GR_I14                       9:0

// Output start value of #14 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR14A_0_DEKNEE_GR_O14                       25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GR14B_0                   (0x3d)
// Slope value of #14 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR14B_0_DEKNEE_GR_S14                       11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_GR15A_0                   (0x3e)
// Input start value of #15 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR15A_0_DEKNEE_GR_I15                       9:0

// Output start value of #15 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR15A_0_DEKNEE_GR_O15                       25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GR15B_0                   (0x3f)
// Slope value of #15 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR15B_0_DEKNEE_GR_S15                       11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_GR16A_0                   (0x40)
// Input start value of #16 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR16A_0_DEKNEE_GR_I16                       9:0

// Output start value of #16 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR16A_0_DEKNEE_GR_O16                       25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GR16B_0                   (0x41)
// Slope value of #16 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GR16B_0_DEKNEE_GR_S16                       11:0


//    (ISP_PIXEL_DIBIT-1):0   rw  DEKNEE_GB_I1    // Input start value of #1 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB1A_0                    (0x42)
// Output start value of #1 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB1A_0_DEKNEE_GB_O1                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GB1B_0                    (0x43)
// Slope value of #1 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB1B_0_DEKNEE_GB_S1                 11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_GB2A_0                    (0x44)
// Input start value of #2 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB2A_0_DEKNEE_GB_I2                 9:0

// Output start value of #2 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB2A_0_DEKNEE_GB_O2                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GB2B_0                    (0x45)
// Slope value of #2 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB2B_0_DEKNEE_GB_S2                 11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_GB3A_0                    (0x46)
// Input start value of #3 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB3A_0_DEKNEE_GB_I3                 9:0

// Output start value of #3 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB3A_0_DEKNEE_GB_O3                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GB3B_0                    (0x47)
// Slope value of #3 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB3B_0_DEKNEE_GB_S3                 11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_GB4A_0                    (0x48)
// Input start value of #4 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB4A_0_DEKNEE_GB_I4                 9:0

// Output start value of #4 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB4A_0_DEKNEE_GB_O4                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GB4B_0                    (0x49)
// Slope value of #4 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB4B_0_DEKNEE_GB_S4                 11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_GB5A_0                    (0x4a)
// Input start value of #5 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB5A_0_DEKNEE_GB_I5                 9:0

// Output start value of #5 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB5A_0_DEKNEE_GB_O5                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GB5B_0                    (0x4b)
// Slope value of #5 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB5B_0_DEKNEE_GB_S5                 11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_GB6A_0                    (0x4c)
// Input start value of #6 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB6A_0_DEKNEE_GB_I6                 9:0

// Output start value of #6 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB6A_0_DEKNEE_GB_O6                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GB6B_0                    (0x4d)
// Slope value of #6 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB6B_0_DEKNEE_GB_S6                 11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_GB7A_0                    (0x4e)
// Input start value of #7 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB7A_0_DEKNEE_GB_I7                 9:0

// Output start value of #7 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB7A_0_DEKNEE_GB_O7                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GB7B_0                    (0x4f)
// Slope value of #7 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB7B_0_DEKNEE_GB_S7                 11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_GB8A_0                    (0x50)
// Input start value of #8 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB8A_0_DEKNEE_GB_I8                 9:0

// Output start value of #8 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB8A_0_DEKNEE_GB_O8                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GB8B_0                    (0x51)
// Slope value of #8 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB8B_0_DEKNEE_GB_S8                 11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_GB9A_0                    (0x52)
// Input start value of #9 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB9A_0_DEKNEE_GB_I9                 9:0

// Output start value of #9 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB9A_0_DEKNEE_GB_O9                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GB9B_0                    (0x53)
// Slope value of #9 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB9B_0_DEKNEE_GB_S9                 11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_GB10A_0                   (0x54)
// Input start value of #10 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB10A_0_DEKNEE_GB_I10                       9:0

// Output start value of #10 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB10A_0_DEKNEE_GB_O10                       25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GB10B_0                   (0x55)
// Slope value of #10 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB10B_0_DEKNEE_GB_S10                       11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_GB11A_0                   (0x56)
// Input start value of #11 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB11A_0_DEKNEE_GB_I11                       9:0

// Output start value of #11 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB11A_0_DEKNEE_GB_O11                       25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GB11B_0                   (0x57)
// Slope value of #11 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB11B_0_DEKNEE_GB_S11                       11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_GB12A_0                   (0x58)
// Input start value of #12 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB12A_0_DEKNEE_GB_I12                       9:0

// Output start value of #12 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB12A_0_DEKNEE_GB_O12                       25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GB12B_0                   (0x59)
// Slope value of #12 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB12B_0_DEKNEE_GB_S12                       11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_GB13A_0                   (0x5a)
// Input start value of #13 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB13A_0_DEKNEE_GB_I13                       9:0

// Output start value of #13 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB13A_0_DEKNEE_GB_O13                       25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GB13B_0                   (0x5b)
// Slope value of #13 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB13B_0_DEKNEE_GB_S13                       11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_GB14A_0                   (0x5c)
// Input start value of #14 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB14A_0_DEKNEE_GB_I14                       9:0

// Output start value of #14 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB14A_0_DEKNEE_GB_O14                       25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GB14B_0                   (0x5d)
// Slope value of #14 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB14B_0_DEKNEE_GB_S14                       11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_GB15A_0                   (0x5e)
// Input start value of #15 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB15A_0_DEKNEE_GB_I15                       9:0

// Output start value of #15 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB15A_0_DEKNEE_GB_O15                       25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GB15B_0                   (0x5f)
// Slope value of #15 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB15B_0_DEKNEE_GB_S15                       11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_GB16A_0                   (0x60)
// Input start value of #16 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB16A_0_DEKNEE_GB_I16                       9:0

// Output start value of #16 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB16A_0_DEKNEE_GB_O16                       25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_GB16B_0                   (0x61)
// Slope value of #16 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_GB16B_0_DEKNEE_GB_S16                       11:0


//    (ISP_PIXEL_DIBIT-1):0   rw  DEKNEE_R_I1     // Input start value of #1 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R1A_0                     (0x62)
// Output start value of #1 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R1A_0_DEKNEE_R_O1                   25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_R1B_0                     (0x63)
// Slope value of #1 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R1B_0_DEKNEE_R_S1                   11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_R2A_0                     (0x64)
// Input start value of #2 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R2A_0_DEKNEE_R_I2                   9:0

// Output start value of #2 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R2A_0_DEKNEE_R_O2                   25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_R2B_0                     (0x65)
// Slope value of #2 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R2B_0_DEKNEE_R_S2                   11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_R3A_0                     (0x66)
// Input start value of #3 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R3A_0_DEKNEE_R_I3                   9:0

// Output start value of #3 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R3A_0_DEKNEE_R_O3                   25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_R3B_0                     (0x67)
// Slope value of #3 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R3B_0_DEKNEE_R_S3                   11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_R4A_0                     (0x68)
// Input start value of #4 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R4A_0_DEKNEE_R_I4                   9:0

// Output start value of #4 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R4A_0_DEKNEE_R_O4                   25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_R4B_0                     (0x69)
// Slope value of #4 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R4B_0_DEKNEE_R_S4                   11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_R5A_0                     (0x6a)
// Input start value of #5 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R5A_0_DEKNEE_R_I5                   9:0

// Output start value of #5 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R5A_0_DEKNEE_R_O5                   25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_R5B_0                     (0x6b)
// Slope value of #5 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R5B_0_DEKNEE_R_S5                   11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_R6A_0                     (0x6c)
// Input start value of #6 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R6A_0_DEKNEE_R_I6                   9:0

// Output start value of #6 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R6A_0_DEKNEE_R_O6                   25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_R6B_0                     (0x6d)
// Slope value of #6 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R6B_0_DEKNEE_R_S6                   11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_R7A_0                     (0x6e)
// Input start value of #7 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R7A_0_DEKNEE_R_I7                   9:0

// Output start value of #7 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R7A_0_DEKNEE_R_O7                   25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_R7B_0                     (0x6f)
// Slope value of #7 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R7B_0_DEKNEE_R_S7                   11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_R8A_0                     (0x70)
// Input start value of #8 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R8A_0_DEKNEE_R_I8                   9:0

// Output start value of #8 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R8A_0_DEKNEE_R_O8                   25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_R8B_0                     (0x71)
// Slope value of #8 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R8B_0_DEKNEE_R_S8                   11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_R9A_0                     (0x72)
// Input start value of #9 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R9A_0_DEKNEE_R_I9                   9:0

// Output start value of #9 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R9A_0_DEKNEE_R_O9                   25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_R9B_0                     (0x73)
// Slope value of #9 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R9B_0_DEKNEE_R_S9                   11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_R10A_0                    (0x74)
// Input start value of #10 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R10A_0_DEKNEE_R_I10                 9:0

// Output start value of #10 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R10A_0_DEKNEE_R_O10                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_R10B_0                    (0x75)
// Slope value of #10 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R10B_0_DEKNEE_R_S10                 11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_R11A_0                    (0x76)
// Input start value of #11 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R11A_0_DEKNEE_R_I11                 9:0

// Output start value of #11 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R11A_0_DEKNEE_R_O11                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_R11B_0                    (0x77)
// Slope value of #11 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R11B_0_DEKNEE_R_S11                 11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_R12A_0                    (0x78)
// Input start value of #12 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R12A_0_DEKNEE_R_I12                 9:0

// Output start value of #12 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R12A_0_DEKNEE_R_O12                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_R12B_0                    (0x79)
// Slope value of #12 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R12B_0_DEKNEE_R_S12                 11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_R13A_0                    (0x7a)
// Input start value of #13 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R13A_0_DEKNEE_R_I13                 9:0

// Output start value of #13 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R13A_0_DEKNEE_R_O13                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_R13B_0                    (0x7b)
// Slope value of #13 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R13B_0_DEKNEE_R_S13                 11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_R14A_0                    (0x7c)
// Input start value of #14 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R14A_0_DEKNEE_R_I14                 9:0

// Output start value of #14 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R14A_0_DEKNEE_R_O14                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_R14B_0                    (0x7d)
// Slope value of #14 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R14B_0_DEKNEE_R_S14                 11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_R15A_0                    (0x7e)
// Input start value of #15 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R15A_0_DEKNEE_R_I15                 9:0

// Output start value of #15 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R15A_0_DEKNEE_R_O15                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_R15B_0                    (0x7f)
// Slope value of #15 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R15B_0_DEKNEE_R_S15                 11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_R16A_0                    (0x80)
// Input start value of #16 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R16A_0_DEKNEE_R_I16                 9:0

// Output start value of #16 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R16A_0_DEKNEE_R_O16                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_R16B_0                    (0x81)
// Slope value of #16 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_R16B_0_DEKNEE_R_S16                 11:0


//    (ISP_PIXEL_DIBIT-1):0   rw  DEKNEE_B_I1     // Input start value of #1 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B1A_0                     (0x82)
// Output start value of #1 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B1A_0_DEKNEE_B_O1                   25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_B1B_0                     (0x83)
// Slope value of #1 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B1B_0_DEKNEE_B_S1                   11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_B2A_0                     (0x84)
// Input start value of #2 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B2A_0_DEKNEE_B_I2                   9:0

// Output start value of #2 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B2A_0_DEKNEE_B_O2                   25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_B2B_0                     (0x85)
// Slope value of #2 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B2B_0_DEKNEE_B_S2                   11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_B3A_0                     (0x86)
// Input start value of #3 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B3A_0_DEKNEE_B_I3                   9:0

// Output start value of #3 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B3A_0_DEKNEE_B_O3                   25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_B3B_0                     (0x87)
// Slope value of #3 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B3B_0_DEKNEE_B_S3                   11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_B4A_0                     (0x88)
// Input start value of #4 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B4A_0_DEKNEE_B_I4                   9:0

// Output start value of #4 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B4A_0_DEKNEE_B_O4                   25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_B4B_0                     (0x89)
// Slope value of #4 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B4B_0_DEKNEE_B_S4                   11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_B5A_0                     (0x8a)
// Input start value of #5 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B5A_0_DEKNEE_B_I5                   9:0

// Output start value of #5 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B5A_0_DEKNEE_B_O5                   25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_B5B_0                     (0x8b)
// Slope value of #5 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B5B_0_DEKNEE_B_S5                   11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_B6A_0                     (0x8c)
// Input start value of #6 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B6A_0_DEKNEE_B_I6                   9:0

// Output start value of #6 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B6A_0_DEKNEE_B_O6                   25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_B6B_0                     (0x8d)
// Slope value of #6 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B6B_0_DEKNEE_B_S6                   11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_B7A_0                     (0x8e)
// Input start value of #7 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B7A_0_DEKNEE_B_I7                   9:0

// Output start value of #7 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B7A_0_DEKNEE_B_O7                   25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_B7B_0                     (0x8f)
// Slope value of #7 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B7B_0_DEKNEE_B_S7                   11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_B8A_0                     (0x90)
// Input start value of #8 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B8A_0_DEKNEE_B_I8                   9:0

// Output start value of #8 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B8A_0_DEKNEE_B_O8                   25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_B8B_0                     (0x91)
// Slope value of #8 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B8B_0_DEKNEE_B_S8                   11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_B9A_0                     (0x92)
// Input start value of #9 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B9A_0_DEKNEE_B_I9                   9:0

// Output start value of #9 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B9A_0_DEKNEE_B_O9                   25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_B9B_0                     (0x93)
// Slope value of #9 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B9B_0_DEKNEE_B_S9                   11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_B10A_0                    (0x94)
// Input start value of #10 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B10A_0_DEKNEE_B_I10                 9:0

// Output start value of #10 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B10A_0_DEKNEE_B_O10                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_B10B_0                    (0x95)
// Slope value of #10 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B10B_0_DEKNEE_B_S10                 11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_B11A_0                    (0x96)
// Input start value of #11 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B11A_0_DEKNEE_B_I11                 9:0

// Output start value of #11 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B11A_0_DEKNEE_B_O11                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_B11B_0                    (0x97)
// Slope value of #11 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B11B_0_DEKNEE_B_S11                 11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_B12A_0                    (0x98)
// Input start value of #12 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B12A_0_DEKNEE_B_I12                 9:0

// Output start value of #12 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B12A_0_DEKNEE_B_O12                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_B12B_0                    (0x99)
// Slope value of #12 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B12B_0_DEKNEE_B_S12                 11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_B13A_0                    (0x9a)
// Input start value of #13 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B13A_0_DEKNEE_B_I13                 9:0

// Output start value of #13 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B13A_0_DEKNEE_B_O13                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_B13B_0                    (0x9b)
// Slope value of #13 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B13B_0_DEKNEE_B_S13                 11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_B14A_0                    (0x9c)
// Input start value of #14 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B14A_0_DEKNEE_B_I14                 9:0

// Output start value of #14 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B14A_0_DEKNEE_B_O14                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_B14B_0                    (0x9d)
// Slope value of #14 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B14B_0_DEKNEE_B_S14                 11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_B15A_0                    (0x9e)
// Input start value of #15 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B15A_0_DEKNEE_B_I15                 9:0

// Output start value of #15 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B15A_0_DEKNEE_B_O15                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_B15B_0                    (0x9f)
// Slope value of #15 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B15B_0_DEKNEE_B_S15                 11:0


#define LWE2B0_ISP_DEKNEE_CONFIG_B16A_0                    (0xa0)
// Input start value of #16 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B16A_0_DEKNEE_B_I16                 9:0

// Output start value of #16 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B16A_0_DEKNEE_B_O16                 25:16


#define LWE2B0_ISP_DEKNEE_CONFIG_B16B_0                    (0xa1)
// Slope value of #16 line segment
#define LWE2B0_ISP_DEKNEE_CONFIG_B16B_0_DEKNEE_B_S16                 11:0

// Lens Shading compensation (new)
// This is implemented as Bezier surfaces for 9 (3x3) patches.
//
// There are 100 control points per color, but to simplify HW,
// the horizontally shared ones are stored twice.  To write
// the control point RAMS, we use an indirect addressing scheme.
// CTRL_PT_OFFSET is the RAM address, CTRL_PT_DATA is 1 control
// point and CTRL_PT_BUFFER is 1 of the 2 RAMs.  When CTRL_PT_DATA
// is written, the RAM is written and the HW assumes CTRL_PT_OFFSET
// and CTRL_PT_BUFFER has been set with the appropriate address. 
// CTRL_PT_OFFSET automatically increments when CTRL_PT_DATA is
// written so you don't have to write CTRL_PT_OFFSET if you're
// writing to the next address.  Likewise, it autoincrement on 
// reads to CTRL_PT_DATA and will return the RAM data at
// CTRL_PT_OFFSET.  If you read CTRL_PT_OFFSET, you'll get back
// the auto-incremented value, not the last written value.
// To simplify buffer synchronization, you should switch the buffer
// setting as soon as you have finished writing that buffer so 
// the next image can use the new settings immediately. 
//
// Writing control points to lens shading RAMs should be in column-major order.
// To write the RAMs, do:
//    reg_wr( CTRL_PT_BUFFER, A ); // Set buffer=a
//    reg_wr( CTRL_PT_OFFSET, 0 ); // Set offset=0
//    for( i=0; i<10; i++ ) { // horizontal direction
//      for j ( j=0; j<10; j++  ) { // vertical direction
//        foreach color ( "r", "gr", "gb", "b" ) {
//          reg_wr( CTRL_PT_DATA, Pcolor,ji );
//          // Internal HW copy of offset autoincrements
//          // after each write
//        }
//        if( j==3 | j==6 ) { // repeat horizontally shared pts
//          foreach color ( "r", "gr", "gb", "b" ) {
//            reg_wr( CTRL_PT_DATA, Pcolor,ji );
//            // Internal HW copy of offset autoincrements
//            // after each write
//          }
//        }
//      }
//    }
//    // Set buffer=b, so A buff can be used (eg read by LS)
//    reg_wr( CTRL_PT_BUFFER, B ); 
//
// Note for the patch dimensions, the following must be true:
//    <PATCH>_WIDTH  >= 12 and an even number
//    <PATCH>_HEIGHT >= 4  and an even number
//

#define LWE2B0_ISP_LS_LPATCH_WIDTH_0                       (0xa2)
#define LWE2B0_ISP_LS_LPATCH_WIDTH_0_L_WIDTH                 13:0


#define LWE2B0_ISP_LS_CPATCH_WIDTH_0                       (0xa3)
#define LWE2B0_ISP_LS_CPATCH_WIDTH_0_C_WIDTH                 13:0


#define LWE2B0_ISP_LS_TPATCH_HEIGHT_0                      (0xa4)
#define LWE2B0_ISP_LS_TPATCH_HEIGHT_0_T_HEIGHT                       13:0


#define LWE2B0_ISP_LS_MPATCH_HEIGHT_0                      (0xa5)
#define LWE2B0_ISP_LS_MPATCH_HEIGHT_0_M_HEIGHT                       13:0


#define LWE2B0_ISP_LS_LPATCH_DELTAU_MSB_0                  (0xa6)
#define LWE2B0_ISP_LS_LPATCH_DELTAU_MSB_0_L_DELTAU_MSB                       2:0


#define LWE2B0_ISP_LS_LPATCH_DELTAU_LSB_0                  (0xa7)
#define LWE2B0_ISP_LS_LPATCH_DELTAU_LSB_0_L_DELTAU_LSB                       31:0


#define LWE2B0_ISP_LS_CPATCH_DELTAU_MSB_0                  (0xa8)
#define LWE2B0_ISP_LS_CPATCH_DELTAU_MSB_0_C_DELTAU_MSB                       2:0


#define LWE2B0_ISP_LS_CPATCH_DELTAU_LSB_0                  (0xa9)
#define LWE2B0_ISP_LS_CPATCH_DELTAU_LSB_0_C_DELTAU_LSB                       31:0


#define LWE2B0_ISP_LS_RPATCH_DELTAU_MSB_0                  (0xaa)
#define LWE2B0_ISP_LS_RPATCH_DELTAU_MSB_0_R_DELTAU_MSB                       2:0


#define LWE2B0_ISP_LS_RPATCH_DELTAU_LSB_0                  (0xab)
#define LWE2B0_ISP_LS_RPATCH_DELTAU_LSB_0_R_DELTAU_LSB                       31:0


#define LWE2B0_ISP_LS_TPATCH_DELTAV_MSB_0                  (0xac)
#define LWE2B0_ISP_LS_TPATCH_DELTAV_MSB_0_T_DELTAV_MSB                       2:0


#define LWE2B0_ISP_LS_TPATCH_DELTAV_LSB_0                  (0xad)
#define LWE2B0_ISP_LS_TPATCH_DELTAV_LSB_0_T_DELTAV_LSB                       31:0


#define LWE2B0_ISP_LS_MPATCH_DELTAV_MSB_0                  (0xae)
#define LWE2B0_ISP_LS_MPATCH_DELTAV_MSB_0_M_DELTAV_MSB                       2:0


#define LWE2B0_ISP_LS_MPATCH_DELTAV_LSB_0                  (0xaf)
#define LWE2B0_ISP_LS_MPATCH_DELTAV_LSB_0_M_DELTAV_LSB                       31:0


#define LWE2B0_ISP_LS_BPATCH_DELTAV_MSB_0                  (0xb0)
#define LWE2B0_ISP_LS_BPATCH_DELTAV_MSB_0_B_DELTAV_MSB                       2:0


#define LWE2B0_ISP_LS_BPATCH_DELTAV_LSB_0                  (0xb1)
#define LWE2B0_ISP_LS_BPATCH_DELTAV_LSB_0_B_DELTAV_LSB                       31:0


#define LWE2B0_ISP_LS_CTRL_PT_OFFSET_0                     (0xb2)
#define LWE2B0_ISP_LS_CTRL_PT_OFFSET_0_LS_CP_OFFSET                  8:0


#define LWE2B0_ISP_LS_CTRL_PT_DATA_0                       (0xb3)
// S4.13 (2's compliment)
#define LWE2B0_ISP_LS_CTRL_PT_DATA_0_LS_CP_DATA                      17:0


#define LWE2B0_ISP_LS_CTRL_PT_BUFFER_0                     (0xb4)
// Other buffer is used by LS processing
#define LWE2B0_ISP_LS_CTRL_PT_BUFFER_0_LS_CP_BUFFER                  0:0
#define LWE2B0_ISP_LS_CTRL_PT_BUFFER_0_LS_CP_BUFFER_A                      (0)
#define LWE2B0_ISP_LS_CTRL_PT_BUFFER_0_LS_CP_BUFFER_B                      (1)

// White Balance gains
// Digital gain values are dedicated to each of the four color channels (R, G1, G2, B).
// If WHITE_BALANCE_MODE in reg ISP_CONTROL1 is set to DYNAMIC, the gain values are updated by
//  the internal AWB control (see regs ISP_WBALANCE_CONTROL1 to ISP_WBALANCE_CONTROL5).
//  Statistics for internal AWB controls are sampled in windows defined by M1 measuring window.
// If WHITE_BALANCE_MODE in reg ISP_CONTROL1 is set to PROGRAMMED, the gain values can be set by
//  Host or AVP. Host or AVP can determine the proper white balance gains based on the 8-bin
//  histogram (see ISP_WBALANCE_CONTROL2), statistics gathered from multiple sub-windows
//  defined in M3 (as many as 64x64 sub-windows, see ISP_TG_CONTROL17), or statistics from M1.

#define LWE2B0_ISP_WBALANCE_PROG1_0                        (0xb5)
// Gain factor for Red
#define LWE2B0_ISP_WBALANCE_PROG1_0_R1WBGAIN                 9:0

// Gain factor for Blue
#define LWE2B0_ISP_WBALANCE_PROG1_0_B1WBGAIN                 25:16


#define LWE2B0_ISP_WBALANCE_PROG2_0                        (0xb6)
// Gain factor for Green on Red
#define LWE2B0_ISP_WBALANCE_PROG2_0_G1WBGAIN                 9:0

// Gain factor for Green on Blue
#define LWE2B0_ISP_WBALANCE_PROG2_0_G2WBGAIN                 25:16

// Auto White Balance (AWB) control
// Internal AWB control determines the proper white balance gains based on the average values
//  and peak values in {R, G1, G2, B} color channels.
// If the frame exposure is high ("Top" level, see ISP_WBALANCE_CONTROL4), gains are callwlated
//  from the average R, G1, G2, and B values;
// If the frame exposure is low ("Bottom" level, see ISP_WBALANCE_CONTROL5), gains are
//  callwlated from the peak R, G1, G2, and B values;
// If the frame exposure is in the middle range, gains are callwlated from both the average
//  values and the peak values.
//
// Top-clip-enable will force top clipping at WB_CLIP_LIMIT regardless of white balance mode.
// When TOP_CLIP_ENABLE = 1, output of WB module is unconditionally clipped to WB_CLIP_LEVEL.

#define LWE2B0_ISP_WBALANCE_CONTROL1_0                     (0xb7)
// Vertical samples for exposure measurement  0 = 32-sample
//  1 = 64-sample
#define LWE2B0_ISP_WBALANCE_CONTROL1_0_LWSAMPLE                      1:1
#define LWE2B0_ISP_WBALANCE_CONTROL1_0_LWSAMPLE_LW32                       (0)
#define LWE2B0_ISP_WBALANCE_CONTROL1_0_LWSAMPLE_LW64                       (1)

// Horizontal samples for exposure measurement  0 = 32-sample
//  1 = 64-sample
#define LWE2B0_ISP_WBALANCE_CONTROL1_0_NHSAMPLE                      2:2
#define LWE2B0_ISP_WBALANCE_CONTROL1_0_NHSAMPLE_NH32                       (0)
#define LWE2B0_ISP_WBALANCE_CONTROL1_0_NHSAMPLE_NH64                       (1)

// Selection of temporal filter coefficients
#define LWE2B0_ISP_WBALANCE_CONTROL1_0_TFCOEF                        5:3

// Maximum limit value of peak detection
#define LWE2B0_ISP_WBALANCE_CONTROL1_0_PEAK_MAX                      25:16


#define LWE2B0_ISP_WBALANCE_CONTROL2_0                     (0xb8)
// Gain-based top-clipping enable  0 = disable
//  1 = enable
#define LWE2B0_ISP_WBALANCE_CONTROL2_0_WB_CLIP_ENABLE                        1:1
#define LWE2B0_ISP_WBALANCE_CONTROL2_0_WB_CLIP_ENABLE_DISABLE                      (0)
#define LWE2B0_ISP_WBALANCE_CONTROL2_0_WB_CLIP_ENABLE_ENABLE                       (1)

// Lower limit of gain-based clipping
#define LWE2B0_ISP_WBALANCE_CONTROL2_0_WB_CLIP_LIMIT                 11:2

// White reference level for gain adjustment
#define LWE2B0_ISP_WBALANCE_CONTROL2_0_WB_PEAK_REF                   25:16


#define LWE2B0_ISP_WBALANCE_CONTROL3_0                     (0xb9)
// Top-to-middle transition level
#define LWE2B0_ISP_WBALANCE_CONTROL3_0_TH_T2M                        9:0

// Middle-to-top transition level
#define LWE2B0_ISP_WBALANCE_CONTROL3_0_TH_M2T                        25:16


#define LWE2B0_ISP_WBALANCE_CONTROL4_0                     (0xba)
// Middle-to-bottom transition level
#define LWE2B0_ISP_WBALANCE_CONTROL4_0_TH_M2B                        9:0

// Bottom-to-middle transition level
#define LWE2B0_ISP_WBALANCE_CONTROL4_0_TH_B2M                        25:16


#define LWE2B0_ISP_WBALANCE_CONTROL5_0                     (0xbb)
// Max gain factor for Red
#define LWE2B0_ISP_WBALANCE_CONTROL5_0_RGAINMAX                      9:0

// Max gain factor for Green
#define LWE2B0_ISP_WBALANCE_CONTROL5_0_GGAINMAX                      19:10

// Max gain factor for Blue
#define LWE2B0_ISP_WBALANCE_CONTROL5_0_BGAINMAX                      29:20


#define LWE2B0_ISP_WBALANCE_CONTROL6_0                     (0xbc)
// Min gain factor for Red
#define LWE2B0_ISP_WBALANCE_CONTROL6_0_RGAINMIN                      9:0

// Min gain factor for Green
#define LWE2B0_ISP_WBALANCE_CONTROL6_0_GGAINMIN                      19:10

// Min gain factor for Blue
#define LWE2B0_ISP_WBALANCE_CONTROL6_0_BGAINMIN                      29:20

// Bad Pixel Concealment
// If a pixel value is highly deviated from its surrounding pixels, the pixel is considered 
// as a bad pixel.  The deviation is defined in terms of the percentage of the average value
// of the pixel surrounding.  Two ratio coefficients are used: if the pixel is at flat area,
// LT_COEF is used, otherwise if it's at edge region, UT_COEF is used. 
// If the average of pixel surround is lower than BP_DARK, it's considered at shadow region,
// and BP_DARK instead of average value will be used to callwlate deviation threshold.    
//
// Noise Reduction
// There are two noise reduction functions in ISP datapath:
// (1) 2D noise reduction with intensity as a factor to sigma filter
//     Noise is reduced not only in image plane but also in intensity domain.  
//     In other words, low pass filtering is applied to pixels that has similar 
//     pixel value and also close to the location.  WA defines the filter kernel 
//     size on the intensity domain.  Larger WA generates smoother but blurry image.  
// (2) False color reduction: For pixels at shadow region or at edges, the colorfulness 
//      is reduced to de-emphasize color noise and color aliasing effect.  
//              MIN_Y and D_TRANSITION specify the pixels that will undergo the color reduction;
//              If the luminance level of the pixel is larger than (MIN_Y + (1<<(D_TRANSITION+4))), 
//      pixel's colorfulness is intact.  If the luminance level is smaller than MIN_Y, 
//              the colorfulness is reduced to minimum (i.e., no color correction applied).  Pixels 
//              with luminance levels in between pass through intermediate colorfulness reduction
//              factor. 
//          CC_HFACTOR specifies how much color reduction should be applied at edge pixels.
//              Larger CC_HFACTOR makes more noticable color reduction at edges. 

#define LWE2B0_ISP_ENHANCE_CONFIG1_0                       (0xbd)
// Bad-pixel dark level
#define LWE2B0_ISP_ENHANCE_CONFIG1_0_BP_DARK                 9:0

// U1.3 ratio for Upper-threshold
#define LWE2B0_ISP_ENHANCE_CONFIG1_0_UT_COEF                 19:16

// U1.3 ratio for Lower-threshold
#define LWE2B0_ISP_ENHANCE_CONFIG1_0_LT_COEF                 23:20

// Weighted average scaling factor in [0,7] 
#define LWE2B0_ISP_ENHANCE_CONFIG1_0_WA                        26:24


#define LWE2B0_ISP_ENHANCE_CONFIG2_0                       (0xbe)
// Color noise reduction, shadow level
#define LWE2B0_ISP_ENHANCE_CONFIG2_0_MIN_Y                   9:0

// Transition width select  0 =  16 levels
//  1 =  32 levels
//  2 =  64 levels
//  3 = 128 levels
#define LWE2B0_ISP_ENHANCE_CONFIG2_0_D_TRANSITION                    17:16
#define LWE2B0_ISP_ENHANCE_CONFIG2_0_D_TRANSITION_D16                      (0)
#define LWE2B0_ISP_ENHANCE_CONFIG2_0_D_TRANSITION_D32                      (1)
#define LWE2B0_ISP_ENHANCE_CONFIG2_0_D_TRANSITION_D64                      (2)
#define LWE2B0_ISP_ENHANCE_CONFIG2_0_D_TRANSITION_D128                     (3)

// Scaling factor in U5.0
#define LWE2B0_ISP_ENHANCE_CONFIG2_0_CC_HFACTOR                      23:19

// Edge Enhancement and Noise Reduction
// ENHLEVEL specifies how heavily the edge enhancement is applied. 
// Edge enhancement is done by (1) extracting high frequency component (HF) of the pixel,
// (2) amplifying the HF if the extracted value is "large" (i.e., coring), and 
// (3) adding the amplified HF back to the Y channel of the pixel.  In this implementation 
// the threshold that's used for coring is not just a pre-determined fixed value, rather 
// it's adaptive depending on the brightness level.  If the pixel surround is bright 
// (more visible), the coring threshold will be larger, which means only stronger edges 
// are enhanced.  Weaker edges are considered more noise-prone and thus not enhanced.
// Two control parameters to determine the adaptive coring threshold:    
// GCORELIMIT specifies the minimum coring level.  The coring level increases when pixel 
// surround's brightness level goes up.
// GCORESCALE specifies the influence of the brightness level to the coring level.
// The higher the GCORESCALE is, the larger the coring level grows under the same brightness.

#define LWE2B0_ISP_ENHANCE_CONFIG3_0                       (0xbf)
// Gain factor of edge enhancement
//  Multiply this value to the edge signal
#define LWE2B0_ISP_ENHANCE_CONFIG3_0_ENHCOEF                 4:0

// Fixed gain applied on top of ENHCOEF  0 = x1 
//  1 = x4 
#define LWE2B0_ISP_ENHANCE_CONFIG3_0_ENH_FIXED_GAIN                  5:5
#define LWE2B0_ISP_ENHANCE_CONFIG3_0_ENH_FIXED_GAIN_X1                     (0)
#define LWE2B0_ISP_ENHANCE_CONFIG3_0_ENH_FIXED_GAIN_X4                     (1)

// Base level of noise coring 
//  
#define LWE2B0_ISP_ENHANCE_CONFIG3_0_GCORELIMIT                      13:8

// Scaling factor range of GCORELIMIT  0 = range [1,4.5]
//  1 = range [1,8.5]
//  2 = range [1,16.5]
//  3 = range [1,32.5]
#define LWE2B0_ISP_ENHANCE_CONFIG3_0_GCORESCALE                      15:14
#define LWE2B0_ISP_ENHANCE_CONFIG3_0_GCORESCALE_CORE4                      (0)
#define LWE2B0_ISP_ENHANCE_CONFIG3_0_GCORESCALE_CORE8                      (1)
#define LWE2B0_ISP_ENHANCE_CONFIG3_0_GCORESCALE_CORE16                     (2)
#define LWE2B0_ISP_ENHANCE_CONFIG3_0_GCORESCALE_CORE32                     (3)

// Programming registers for demosaicing formulas
// Programmable 5x5 demosaicing takes 56 bits per formula.
//   25 pixels are divided into 8 groups. 
//   8 coefficients are made programmable.
// For 3-color CFA, e.g. RGB Bayer, there are 6 formulas.
//   2 formulas for Green. {1G and 2G}
//   4 formulas shared for Red and Blue. {1C, 2C, 3C, 4C}
// For 4-color CFA, e.g. CMYK and RGBY, there are 4 formulas.
//   4 formulas shared by all 4 colors.  {1C, 2C, 3C, 4C}
// There are 6 sets of 2-register pairs.
//   Each pair of registers holds 56 programming bits of one formula.
//     p_value[56:0] = {regX2[24:0], regX1[31:0]}
//   Formula:1G: {dm_coef12g[24:0], dm_coef11g[31:0]}   
//   Formula:2G: {dm_coef22g[24:0], dm_coef21g[31:0]}   
//   Formula:1C: {dm_coef12c[24:0], dm_coef11c[31:0]}   
//   Formula:2C: {dm_coef22c[24:0], dm_coef21c[31:0]}   
//   Formula:3C: {dm_coef32c[24:0], dm_coef31c[31:0]}   
//   Formula:4C: {dm_coef42c[24:0], dm_coef41c[31:0]}
//
// Note that all the programming bits are encrypted.
// SC_SEED[6:0] is seed value for the encryption.

#define LWE2B0_ISP_DM_FORMULA_G11_0                        (0xc0)
// dm_coef11g                   
#define LWE2B0_ISP_DM_FORMULA_G11_0_DM_COEF11G                       31:0


#define LWE2B0_ISP_DM_FORMULA_G12_0                        (0xc1)
// dm_coef12g 
#define LWE2B0_ISP_DM_FORMULA_G12_0_DM_COEF12G                       24:0

// Seed value in {0, 127}                  
#define LWE2B0_ISP_DM_FORMULA_G12_0_SC_SEED                  31:25


#define LWE2B0_ISP_DM_FORMULA_G21_0                        (0xc2)
// dm_coef21g                   
#define LWE2B0_ISP_DM_FORMULA_G21_0_DM_COEF21G                       31:0


#define LWE2B0_ISP_DM_FORMULA_G22_0                        (0xc3)
// dm_coef22g                   
#define LWE2B0_ISP_DM_FORMULA_G22_0_DM_COEF22G                       24:0


#define LWE2B0_ISP_DM_FORMULA_C11_0                        (0xc4)
// dm_coef11c                   
#define LWE2B0_ISP_DM_FORMULA_C11_0_DM_COEF11C                       31:0


#define LWE2B0_ISP_DM_FORMULA_C12_0                        (0xc5)
// dm_coef12c                   
#define LWE2B0_ISP_DM_FORMULA_C12_0_DM_COEF12C                       24:0


#define LWE2B0_ISP_DM_FORMULA_C21_0                        (0xc6)
// dm_coef21c                   
#define LWE2B0_ISP_DM_FORMULA_C21_0_DM_COEF21C                       31:0


#define LWE2B0_ISP_DM_FORMULA_C22_0                        (0xc7)
// dm_coef22c                   
#define LWE2B0_ISP_DM_FORMULA_C22_0_DM_COEF22C                       24:0


#define LWE2B0_ISP_DM_FORMULA_C31_0                        (0xc8)
// dm_coef31c                   
#define LWE2B0_ISP_DM_FORMULA_C31_0_DM_COEF31C                       31:0


#define LWE2B0_ISP_DM_FORMULA_C32_0                        (0xc9)
// dm_coef32c                   
#define LWE2B0_ISP_DM_FORMULA_C32_0_DM_COEF32C                       24:0


#define LWE2B0_ISP_DM_FORMULA_C41_0                        (0xca)
// dm_coef41c                   
#define LWE2B0_ISP_DM_FORMULA_C41_0_DM_COEF41C                       31:0


#define LWE2B0_ISP_DM_FORMULA_C42_0                        (0xcb)
// dm_coef42c                   
#define LWE2B0_ISP_DM_FORMULA_C42_0_DM_COEF42C                       24:0

// Color artifact reduction
//
// "car_enable" bit switches color-artifact reduction processing on and off
// For layer L1 image,
//   "car_w1" specifies group size of pixels of the same color that are summed up
//      for callwlation of average values 
//   "car_vf1k specifies filter coefficient for the input of vertical IIR filter
//   "car_dif1core" specifies noise coring level to the L1 weighting signal
//   "car_f1mag" specifies magnifying factor to the L1 weighting factor
// For layer L2 image,
//   "car_w2" specifies group size of pixels of the same color that are summed up 
//      for callwlation of average values 
//   "car_vf2k specifies filter coefficient for the input of vertical IIR filter
//   "car_dif2core" specifies noise coring level to the L2 weighting signal
//   "car_f2mag" specifies magnifying factor to the L2 weighting factor

#define LWE2B0_ISP_CAR_CONTROL1_0                  (0xcc)
// Group size  0 =  2 pixel
//  1 =  4 pixel
//  2 =  8 pixel
//  3 =  16 pixel
//  4 =  32 pixel
//  5 =  64 pixel
//  6 =  128 pixel
#define LWE2B0_ISP_CAR_CONTROL1_0_CAR_W1                     2:0
#define LWE2B0_ISP_CAR_CONTROL1_0_CAR_W1_G2                        (0)
#define LWE2B0_ISP_CAR_CONTROL1_0_CAR_W1_G4                        (1)
#define LWE2B0_ISP_CAR_CONTROL1_0_CAR_W1_G8                        (2)
#define LWE2B0_ISP_CAR_CONTROL1_0_CAR_W1_G16                       (3)
#define LWE2B0_ISP_CAR_CONTROL1_0_CAR_W1_G32                       (4)
#define LWE2B0_ISP_CAR_CONTROL1_0_CAR_W1_G64                       (5)
#define LWE2B0_ISP_CAR_CONTROL1_0_CAR_W1_G128                      (6)

// U1.4, max value is 16 (mathematically 1.0). 
#define LWE2B0_ISP_CAR_CONTROL1_0_CAR_VF1K                   7:3

// U8.0
#define LWE2B0_ISP_CAR_CONTROL1_0_CAR_DIF1CORE                       15:8

// U4.4
#define LWE2B0_ISP_CAR_CONTROL1_0_CAR_F1MAG                  23:16


#define LWE2B0_ISP_CAR_CONTROL2_0                  (0xcd)
// Group size  0 =  2 pixel
//  1 =  4 pixel
//  2 =  8 pixel
//  3 =  16 pixel
//  4 =  32 pixel
//  5 =  64 pixel
//  6 =  128 pixel
#define LWE2B0_ISP_CAR_CONTROL2_0_CAR_W2                     2:0
#define LWE2B0_ISP_CAR_CONTROL2_0_CAR_W2_G2                        (0)
#define LWE2B0_ISP_CAR_CONTROL2_0_CAR_W2_G4                        (1)
#define LWE2B0_ISP_CAR_CONTROL2_0_CAR_W2_G8                        (2)
#define LWE2B0_ISP_CAR_CONTROL2_0_CAR_W2_G16                       (3)
#define LWE2B0_ISP_CAR_CONTROL2_0_CAR_W2_G32                       (4)
#define LWE2B0_ISP_CAR_CONTROL2_0_CAR_W2_G64                       (5)
#define LWE2B0_ISP_CAR_CONTROL2_0_CAR_W2_G128                      (6)

// U1.4, max value is 16 (mathematically 1.0).
#define LWE2B0_ISP_CAR_CONTROL2_0_CAR_VF2K                   7:3

// U8.0
#define LWE2B0_ISP_CAR_CONTROL2_0_CAR_DIF2CORE                       15:8

// U4.4
#define LWE2B0_ISP_CAR_CONTROL2_0_CAR_F2MAG                  23:16

// Image Down-Scaling
// Only down-scaling is available.
// Horizontal and vertical down-scaling ratios are treated independently.
// Two DDAs are provided for horizontal and vertical timing control.
// Both DDAs have 6-bit integer and 10-bit fraction.
//   That can handle up to 63-to-1 down-scaling.
//   A DDA is configured by {initial_value, increment_value}.
//   "Initial_value" specifies relative positions of first active lines in the 
//   source image and the destination image.
//     This is represented in 8-bit, U3.5.
//   "increment_value" is the sampling pitch of the destination image measured in the 
//   sampling pitch of the source image.
//
// For the vertical down-scaling,
//   Initial_value = 0.0 means that the two first active lines of source and destination 
//   images are the same position.
//   Increment_value = 2.5 for example means 2.5-to-1 down-scaling.
//   Increment_value >= 1.0. (down-scaling only)
// For the horizontal down-scaling,
//   Initial_value can be in [0.0, 7.96875].
//   Initial_value = 0.0 means that the two first active pixels of source and destination 
//   images are the same position.
//   Increment_value = 10.5 for example means 10.5-to-1 down-scaling.
//   Increment_value >= 1.0. (down-scaling only)
//
// To make the destination image exactly TH lines by TW pixels, image height (lines) and
// image width (pixels) are specified.
// Exactly THxTW destination image will be produced if and only if  the specified size of 
// a destination image fits in the destination image frame specified by {source image size, 
// H & V DDA configurations}.
// If not, size of the destination image will be limited by the size of source image.

#define LWE2B0_ISP_DS_VERTICAL_0                   (0xce)
// Increment_value, U6.10
//  DS_VERT_DELTA=2.5 means 2.5 to 1 downscaling.
//  Must be >= 1.0 (i.e., downscaling only)
#define LWE2B0_ISP_DS_VERTICAL_0_DS_VERT_DELTA                       15:0

// Down-scaler enable
//  0    Disable (default)
//  1    Enable
#define LWE2B0_ISP_DS_VERTICAL_0_DSCALE_INCLUDE                      16:16

// Initial_value, U3.5 
//  Relative position of first active line in the 
//  source image to the destination image.                  
#define LWE2B0_ISP_DS_VERTICAL_0_DS_VERT_INITIAL                     31:24

#define LWE2B0_ISP_DS_HORIZONTAL_0                 (0xcf)
// Increment_value, U6.10 
//  DS_HOR_DELTA=2.5 means 2.5 to 1 downscaling.
//  Must be >= 1.0 (i.e., downscaling only)
#define LWE2B0_ISP_DS_HORIZONTAL_0_DS_HOR_DELTA                      15:0

// Selection of horizontal low-pass filters 
//  in {0,1,2,3,4,5,6,7}
#define LWE2B0_ISP_DS_HORIZONTAL_0_DS_HOR_FILTER                     18:16

// Initial_value, U3.5                    
//  Relative position of first active pixel in the 
//  source image to the destination image.                  
#define LWE2B0_ISP_DS_HORIZONTAL_0_DS_HOR_INITIAL                    31:24


#define LWE2B0_ISP_DS_DEST_SIZE_0                  (0xd0)
// Number of active pixels in a line 
#define LWE2B0_ISP_DS_DEST_SIZE_0_DS_DEST_WIDTH                      13:0

// Number of active lines in a frame
#define LWE2B0_ISP_DS_DEST_SIZE_0_DS_DEST_HEIGHT                     29:16

// Color Correction
// 3x3 matrix factors for the camera RGB (cRGB) to standard RGB (sRGB) colwersion.
// Subjective adjustment of color may also be included.

#define LWE2B0_ISP_COLORCORRECT_RCONFIG1_0                 (0xd1)
// cR to sR factor, pos/neg
#define LWE2B0_ISP_COLORCORRECT_RCONFIG1_0_CCR2R                     11:0

// cR to sG factor, pos/neg
#define LWE2B0_ISP_COLORCORRECT_RCONFIG1_0_CCR2G                     27:16


#define LWE2B0_ISP_COLORCORRECT_RCONFIG2_0                 (0xd2)
// cR to sB factor, pos/neg
#define LWE2B0_ISP_COLORCORRECT_RCONFIG2_0_CCR2B                     11:0


#define LWE2B0_ISP_COLORCORRECT_GCONFIG1_0                 (0xd3)
// cG to sG factor, pos/neg
#define LWE2B0_ISP_COLORCORRECT_GCONFIG1_0_CCG2G                     11:0

// cG to sR factor, pos/neg
#define LWE2B0_ISP_COLORCORRECT_GCONFIG1_0_CCG2R                     27:16


#define LWE2B0_ISP_COLORCORRECT_GCONFIG2_0                 (0xd4)
// cG to sB factor, pos/neg
#define LWE2B0_ISP_COLORCORRECT_GCONFIG2_0_CCG2B                     11:0


#define LWE2B0_ISP_COLORCORRECT_BCONFIG1_0                 (0xd5)
// cB to sB factor, pos/neg
#define LWE2B0_ISP_COLORCORRECT_BCONFIG1_0_CCB2B                     11:0

// cB to sR factor, pos/neg
#define LWE2B0_ISP_COLORCORRECT_BCONFIG1_0_CCB2R                     27:16


#define LWE2B0_ISP_COLORCORRECT_BCONFIG2_0                 (0xd6)
// cB to sG factor, pos/neg
#define LWE2B0_ISP_COLORCORRECT_BCONFIG2_0_CCB2G                     11:0

// Gamma Correction
// Piecewise linear approximation of up to 32 line segments can be configured.
// Line segment definitions must be provided in ascending order (of input range) from CONFIG1
//   to CONFIG32. 
// Unused configuration registers must have "input start value" equals to "maximum input value"
//   and "output start value" equals to "output value to the maximum input value".
// If input start value of the first segment (GAMMA_I1) is not ZERO, input below this value
//   is clumped to this level. (Corresponding output is GAMMA_O1)

#define LWE2B0_ISP_GAMMA_CONFIG1_0                 (0xd7)
// Input start value of #1 line segment
#define LWE2B0_ISP_GAMMA_CONFIG1_0_GAMMA_I1                  9:0

// Output start value of #1 line segment
#define LWE2B0_ISP_GAMMA_CONFIG1_0_GAMMA_O1                  17:10

// Slope value of #1 line segment
#define LWE2B0_ISP_GAMMA_CONFIG1_0_GAMMA_S1                  31:20


#define LWE2B0_ISP_GAMMA_CONFIG2_0                 (0xd8)
// Input start value of #2 line segment
#define LWE2B0_ISP_GAMMA_CONFIG2_0_GAMMA_I2                  9:0

// Output start value of #2 line segment
#define LWE2B0_ISP_GAMMA_CONFIG2_0_GAMMA_O2                  17:10

// Slope value of #2 line segment
#define LWE2B0_ISP_GAMMA_CONFIG2_0_GAMMA_S2                  31:20


#define LWE2B0_ISP_GAMMA_CONFIG3_0                 (0xd9)
// Input start value of #3 line segment
#define LWE2B0_ISP_GAMMA_CONFIG3_0_GAMMA_I3                  9:0

// Output start value of #3 line segment
#define LWE2B0_ISP_GAMMA_CONFIG3_0_GAMMA_O3                  17:10

// Slope value of #3 line segment
#define LWE2B0_ISP_GAMMA_CONFIG3_0_GAMMA_S3                  31:20


#define LWE2B0_ISP_GAMMA_CONFIG4_0                 (0xda)
// Input start value of #4 line segment
#define LWE2B0_ISP_GAMMA_CONFIG4_0_GAMMA_I4                  9:0

// Output start value of #4 line segment
#define LWE2B0_ISP_GAMMA_CONFIG4_0_GAMMA_O4                  17:10

// Slope value of #4 line segment
#define LWE2B0_ISP_GAMMA_CONFIG4_0_GAMMA_S4                  31:20


#define LWE2B0_ISP_GAMMA_CONFIG5_0                 (0xdb)
// Input start value of #5 line segment
#define LWE2B0_ISP_GAMMA_CONFIG5_0_GAMMA_I5                  9:0

// Output start value of #5 line segment
#define LWE2B0_ISP_GAMMA_CONFIG5_0_GAMMA_O5                  17:10

// Slope value of #5 line segment
#define LWE2B0_ISP_GAMMA_CONFIG5_0_GAMMA_S5                  31:20


#define LWE2B0_ISP_GAMMA_CONFIG6_0                 (0xdc)
// Input start value of #6 line segment
#define LWE2B0_ISP_GAMMA_CONFIG6_0_GAMMA_I6                  9:0

// Output start value of #6 line segment
#define LWE2B0_ISP_GAMMA_CONFIG6_0_GAMMA_O6                  17:10

// Slope value of #6 line segment
#define LWE2B0_ISP_GAMMA_CONFIG6_0_GAMMA_S6                  31:20


#define LWE2B0_ISP_GAMMA_CONFIG7_0                 (0xdd)
// Input start value of #7 line segment
#define LWE2B0_ISP_GAMMA_CONFIG7_0_GAMMA_I7                  9:0

// Output start value of #7 line segment
#define LWE2B0_ISP_GAMMA_CONFIG7_0_GAMMA_O7                  17:10

// Slope value of #7 line segment
#define LWE2B0_ISP_GAMMA_CONFIG7_0_GAMMA_S7                  31:20


#define LWE2B0_ISP_GAMMA_CONFIG8_0                 (0xde)
// Input start value of #8 line segment
#define LWE2B0_ISP_GAMMA_CONFIG8_0_GAMMA_I8                  9:0

// Output start value of #8 line segment
#define LWE2B0_ISP_GAMMA_CONFIG8_0_GAMMA_O8                  17:10

// Slope value of #8 line segment
#define LWE2B0_ISP_GAMMA_CONFIG8_0_GAMMA_S8                  31:20


#define LWE2B0_ISP_GAMMA_CONFIG9_0                 (0xdf)
// Input start value of #9 line segment
#define LWE2B0_ISP_GAMMA_CONFIG9_0_GAMMA_I9                  9:0

// Output start value of #9 line segment
#define LWE2B0_ISP_GAMMA_CONFIG9_0_GAMMA_O9                  17:10

// Slope value of #9 line segment
#define LWE2B0_ISP_GAMMA_CONFIG9_0_GAMMA_S9                  31:20


#define LWE2B0_ISP_GAMMA_CONFIG10_0                        (0xe0)
// Input start value of #10 line segment
#define LWE2B0_ISP_GAMMA_CONFIG10_0_GAMMA_I10                        9:0

// Output start value of #10 line segment
#define LWE2B0_ISP_GAMMA_CONFIG10_0_GAMMA_O10                        17:10

// Slope value of #10 line segment
#define LWE2B0_ISP_GAMMA_CONFIG10_0_GAMMA_S10                        31:20


#define LWE2B0_ISP_GAMMA_CONFIG11_0                        (0xe1)
// Input start value of #11 line segment
#define LWE2B0_ISP_GAMMA_CONFIG11_0_GAMMA_I11                        9:0

// Output start value of #11 line segment
#define LWE2B0_ISP_GAMMA_CONFIG11_0_GAMMA_O11                        17:10

// Slope value of #11 line segment
#define LWE2B0_ISP_GAMMA_CONFIG11_0_GAMMA_S11                        31:20


#define LWE2B0_ISP_GAMMA_CONFIG12_0                        (0xe2)
// Input start value of #12 line segment
#define LWE2B0_ISP_GAMMA_CONFIG12_0_GAMMA_I12                        9:0

// Output start value of #12 line segment
#define LWE2B0_ISP_GAMMA_CONFIG12_0_GAMMA_O12                        17:10

// Slope value of #12 line segment
#define LWE2B0_ISP_GAMMA_CONFIG12_0_GAMMA_S12                        31:20


#define LWE2B0_ISP_GAMMA_CONFIG13_0                        (0xe3)
// Input start value of #13 line segment
#define LWE2B0_ISP_GAMMA_CONFIG13_0_GAMMA_I13                        9:0

// Output start value of #13 line segment
#define LWE2B0_ISP_GAMMA_CONFIG13_0_GAMMA_O13                        17:10

// Slope value of #13 line segment
#define LWE2B0_ISP_GAMMA_CONFIG13_0_GAMMA_S13                        31:20


#define LWE2B0_ISP_GAMMA_CONFIG14_0                        (0xe4)
// Input start value of #14 line segment
#define LWE2B0_ISP_GAMMA_CONFIG14_0_GAMMA_I14                        9:0

// Output start value of #14 line segment
#define LWE2B0_ISP_GAMMA_CONFIG14_0_GAMMA_O14                        17:10

// Slope value of #14 line segment
#define LWE2B0_ISP_GAMMA_CONFIG14_0_GAMMA_S14                        31:20


#define LWE2B0_ISP_GAMMA_CONFIG15_0                        (0xe5)
// Input start value of #15 line segment
#define LWE2B0_ISP_GAMMA_CONFIG15_0_GAMMA_I15                        9:0

// Output start value of #15 line segment
#define LWE2B0_ISP_GAMMA_CONFIG15_0_GAMMA_O15                        17:10

// Slope value of #15 line segment
#define LWE2B0_ISP_GAMMA_CONFIG15_0_GAMMA_S15                        31:20


#define LWE2B0_ISP_GAMMA_CONFIG16_0                        (0xe6)
// Input start value of #16 line segment
#define LWE2B0_ISP_GAMMA_CONFIG16_0_GAMMA_I16                        9:0

// Output start value of #16 line segment
#define LWE2B0_ISP_GAMMA_CONFIG16_0_GAMMA_O16                        17:10

// Slope value of #16 line segment
#define LWE2B0_ISP_GAMMA_CONFIG16_0_GAMMA_S16                        31:20


#define LWE2B0_ISP_GAMMA_CONFIG17_0                        (0xe7)
// Input start value of #17 line segment
#define LWE2B0_ISP_GAMMA_CONFIG17_0_GAMMA_I17                        9:0

// Output start value of #17 line segment
#define LWE2B0_ISP_GAMMA_CONFIG17_0_GAMMA_O17                        17:10

// Slope value of #17 line segment
#define LWE2B0_ISP_GAMMA_CONFIG17_0_GAMMA_S17                        31:20


#define LWE2B0_ISP_GAMMA_CONFIG18_0                        (0xe8)
// Input start value of #18 line segment
#define LWE2B0_ISP_GAMMA_CONFIG18_0_GAMMA_I18                        9:0

// Output start value of #18 line segment
#define LWE2B0_ISP_GAMMA_CONFIG18_0_GAMMA_O18                        17:10

// Slope value of #18 line segment
#define LWE2B0_ISP_GAMMA_CONFIG18_0_GAMMA_S18                        31:20


#define LWE2B0_ISP_GAMMA_CONFIG19_0                        (0xe9)
// Input start value of #19 line segment
#define LWE2B0_ISP_GAMMA_CONFIG19_0_GAMMA_I19                        9:0

// Output start value of #19 line segment
#define LWE2B0_ISP_GAMMA_CONFIG19_0_GAMMA_O19                        17:10

// Slope value of #19 line segment
#define LWE2B0_ISP_GAMMA_CONFIG19_0_GAMMA_S19                        31:20


#define LWE2B0_ISP_GAMMA_CONFIG20_0                        (0xea)
// Input start value of #20 line segment
#define LWE2B0_ISP_GAMMA_CONFIG20_0_GAMMA_I20                        9:0

// Output start value of #20 line segment
#define LWE2B0_ISP_GAMMA_CONFIG20_0_GAMMA_O20                        17:10

// Slope value of #20 line segment
#define LWE2B0_ISP_GAMMA_CONFIG20_0_GAMMA_S20                        31:20


#define LWE2B0_ISP_GAMMA_CONFIG21_0                        (0xeb)
// Input start value of #21 line segment
#define LWE2B0_ISP_GAMMA_CONFIG21_0_GAMMA_I21                        9:0

// Output start value of #21 line segment
#define LWE2B0_ISP_GAMMA_CONFIG21_0_GAMMA_O21                        17:10

// Slope value of #21 line segment
#define LWE2B0_ISP_GAMMA_CONFIG21_0_GAMMA_S21                        31:20


#define LWE2B0_ISP_GAMMA_CONFIG22_0                        (0xec)
// Input start value of #22 line segment
#define LWE2B0_ISP_GAMMA_CONFIG22_0_GAMMA_I22                        9:0

// Output start value of #22 line segment
#define LWE2B0_ISP_GAMMA_CONFIG22_0_GAMMA_O22                        17:10

// Slope value of #22 line segment
#define LWE2B0_ISP_GAMMA_CONFIG22_0_GAMMA_S22                        31:20


#define LWE2B0_ISP_GAMMA_CONFIG23_0                        (0xed)
// Input start value of #23 line segment
#define LWE2B0_ISP_GAMMA_CONFIG23_0_GAMMA_I23                        9:0

// Output start value of #23 line segment
#define LWE2B0_ISP_GAMMA_CONFIG23_0_GAMMA_O23                        17:10

// Slope value of #23 line segment
#define LWE2B0_ISP_GAMMA_CONFIG23_0_GAMMA_S23                        31:20


#define LWE2B0_ISP_GAMMA_CONFIG24_0                        (0xee)
// Input start value of #24 line segment
#define LWE2B0_ISP_GAMMA_CONFIG24_0_GAMMA_I24                        9:0

// Output start value of #24 line segment
#define LWE2B0_ISP_GAMMA_CONFIG24_0_GAMMA_O24                        17:10

// Slope value of #24 line segment
#define LWE2B0_ISP_GAMMA_CONFIG24_0_GAMMA_S24                        31:20


#define LWE2B0_ISP_GAMMA_CONFIG25_0                        (0xef)
// Input start value of #25 line segment
#define LWE2B0_ISP_GAMMA_CONFIG25_0_GAMMA_I25                        9:0

// Output start value of #25 line segment
#define LWE2B0_ISP_GAMMA_CONFIG25_0_GAMMA_O25                        17:10

// Slope value of #25 line segment
#define LWE2B0_ISP_GAMMA_CONFIG25_0_GAMMA_S25                        31:20


#define LWE2B0_ISP_GAMMA_CONFIG26_0                        (0xf0)
// Input start value of #26 line segment
#define LWE2B0_ISP_GAMMA_CONFIG26_0_GAMMA_I26                        9:0

// Output start value of #26 line segment
#define LWE2B0_ISP_GAMMA_CONFIG26_0_GAMMA_O26                        17:10

// Slope value of #26 line segment
#define LWE2B0_ISP_GAMMA_CONFIG26_0_GAMMA_S26                        31:20


#define LWE2B0_ISP_GAMMA_CONFIG27_0                        (0xf1)
// Input start value of #27 line segment
#define LWE2B0_ISP_GAMMA_CONFIG27_0_GAMMA_I27                        9:0

// Output start value of #27 line segment
#define LWE2B0_ISP_GAMMA_CONFIG27_0_GAMMA_O27                        17:10

// Slope value of #27 line segment
#define LWE2B0_ISP_GAMMA_CONFIG27_0_GAMMA_S27                        31:20


#define LWE2B0_ISP_GAMMA_CONFIG28_0                        (0xf2)
// Input start value of #28 line segment
#define LWE2B0_ISP_GAMMA_CONFIG28_0_GAMMA_I28                        9:0

// Output start value of #28 line segment
#define LWE2B0_ISP_GAMMA_CONFIG28_0_GAMMA_O28                        17:10

// Slope value of #28 line segment
#define LWE2B0_ISP_GAMMA_CONFIG28_0_GAMMA_S28                        31:20


#define LWE2B0_ISP_GAMMA_CONFIG29_0                        (0xf3)
// Input start value of #29 line segment
#define LWE2B0_ISP_GAMMA_CONFIG29_0_GAMMA_I29                        9:0

// Output start value of #29 line segment
#define LWE2B0_ISP_GAMMA_CONFIG29_0_GAMMA_O29                        17:10

// Slope value of #29 line segment
#define LWE2B0_ISP_GAMMA_CONFIG29_0_GAMMA_S29                        31:20


#define LWE2B0_ISP_GAMMA_CONFIG30_0                        (0xf4)
// Input start value of #30 line segment
#define LWE2B0_ISP_GAMMA_CONFIG30_0_GAMMA_I30                        9:0

// Output start value of #30 line segment
#define LWE2B0_ISP_GAMMA_CONFIG30_0_GAMMA_O30                        17:10

// Slope value of #30 line segment
#define LWE2B0_ISP_GAMMA_CONFIG30_0_GAMMA_S30                        31:20


#define LWE2B0_ISP_GAMMA_CONFIG31_0                        (0xf5)
// Input start value of #31 line segment
#define LWE2B0_ISP_GAMMA_CONFIG31_0_GAMMA_I31                        9:0

// Output start value of #31 line segment
#define LWE2B0_ISP_GAMMA_CONFIG31_0_GAMMA_O31                        17:10

// Slope value of #31 line segment
#define LWE2B0_ISP_GAMMA_CONFIG31_0_GAMMA_S31                        31:20


#define LWE2B0_ISP_GAMMA_CONFIG32_0                        (0xf6)
// Input start value of #32 line segment
#define LWE2B0_ISP_GAMMA_CONFIG32_0_GAMMA_I32                        9:0

// Output start value of #32 line segment
#define LWE2B0_ISP_GAMMA_CONFIG32_0_GAMMA_O32                        17:10

// Slope value of #32 line segment
#define LWE2B0_ISP_GAMMA_CONFIG32_0_GAMMA_S32                        31:20

// RGB to YUV colwersion and Y/C adjustment on Y/U/V
// This is implemented as 3x3 matrix multiplication

#define LWE2B0_ISP_CSC_YCONFIG1_0                  (0xf7)
// Factor for Red to Y
#define LWE2B0_ISP_CSC_YCONFIG1_0_CSCR2Y                     7:0

// Factor for Green to Y
#define LWE2B0_ISP_CSC_YCONFIG1_0_CSCG2Y                     18:10

// Factor for Blue to Y
#define LWE2B0_ISP_CSC_YCONFIG1_0_CSCB2Y                     27:20


#define LWE2B0_ISP_CSC_YCONFIG2_0                  (0xf8)
// Offset adjust for Y
#define LWE2B0_ISP_CSC_YCONFIG2_0_CSCYOFF                    7:0

// Signal range of YUV outputs   0 = [0,255]
//  1 = [1,254] exclude 00 and FF
//  2 = ITU601 standard
//      Y : [16,235]
//      UV: [16,240]
#define LWE2B0_ISP_CSC_YCONFIG2_0_YUVRANGE                   17:16
#define LWE2B0_ISP_CSC_YCONFIG2_0_YUVRANGE_FULL                    (0)
#define LWE2B0_ISP_CSC_YCONFIG2_0_YUVRANGE_NEARFULL                        (1)
#define LWE2B0_ISP_CSC_YCONFIG2_0_YUVRANGE_ITU601                  (2)


#define LWE2B0_ISP_CSC_UCONFIG_0                   (0xf9)
// Factor for Red to U
#define LWE2B0_ISP_CSC_UCONFIG_0_CSCR2U                      8:0

// Factor for Green to U
#define LWE2B0_ISP_CSC_UCONFIG_0_CSCG2U                      18:10

// Factor for Blue to U
#define LWE2B0_ISP_CSC_UCONFIG_0_CSCB2U                      28:20


#define LWE2B0_ISP_CSC_VCONFIG_0                   (0xfa)
// Factor for Red to V
#define LWE2B0_ISP_CSC_VCONFIG_0_CSCR2V                      8:0

// Factor for Green to V
#define LWE2B0_ISP_CSC_VCONFIG_0_CSCG2V                      18:10

// Factor for Blue to V
#define LWE2B0_ISP_CSC_VCONFIG_0_CSCB2V                      28:20

// U and V offset control for CSC
// Positive/negative 2's complement values are added to U and V outputs.

#define LWE2B0_ISP_CSC_UVOFFSETCONFIG_0                    (0xfb)
// Offset value for U, pos/neg (DBIT).0
#define LWE2B0_ISP_CSC_UVOFFSETCONFIG_0_UOFFSET                      7:0

// Offset value for V, pos/neg (DBIT).0
#define LWE2B0_ISP_CSC_UVOFFSETCONFIG_0_VOFFSET                      23:16

// Special effects
// Solarization (posterization)
// To manipulate pixel data bits individually.
// To be operated on R/G/B pixels after gamma correction.
// Number of bits per color component is ISP_GAMMA_DOBIT.
// Each bit (from LSB to MSB) is individually colwerted in four ways.
//   code    input    output
//    00       S        0      (forced to 0)
//    01       S        1      (forced to 1)
//    10       S        S      (as is)
//    11       S        !S     (ilwerted)

#define LWE2B0_ISP_SOLARIZE_CONTROL1_0                     (0xfc)
// Red, LSB
#define LWE2B0_ISP_SOLARIZE_CONTROL1_0_SOL_R0                        1:0

// Red, LSB+1
#define LWE2B0_ISP_SOLARIZE_CONTROL1_0_SOL_R1                        3:2

// Red, LSB+2
#define LWE2B0_ISP_SOLARIZE_CONTROL1_0_SOL_R2                        5:4

// Red, LSB+3
#define LWE2B0_ISP_SOLARIZE_CONTROL1_0_SOL_R3                        7:6

// Red, LSB+4
#define LWE2B0_ISP_SOLARIZE_CONTROL1_0_SOL_R4                        9:8

// Red, LSB+5
#define LWE2B0_ISP_SOLARIZE_CONTROL1_0_SOL_R5                        11:10

// Red, LSB+6
#define LWE2B0_ISP_SOLARIZE_CONTROL1_0_SOL_R6                        13:12

// Red, LSB+7
#define LWE2B0_ISP_SOLARIZE_CONTROL1_0_SOL_R7                        15:14

// Green, LSB
#define LWE2B0_ISP_SOLARIZE_CONTROL1_0_SOL_G0                        17:16

// Green, LSB+1
#define LWE2B0_ISP_SOLARIZE_CONTROL1_0_SOL_G1                        19:18

// Green, LSB+2
#define LWE2B0_ISP_SOLARIZE_CONTROL1_0_SOL_G2                        21:20

// Green, LSB+3
#define LWE2B0_ISP_SOLARIZE_CONTROL1_0_SOL_G3                        23:22

// Green, LSB+4
#define LWE2B0_ISP_SOLARIZE_CONTROL1_0_SOL_G4                        25:24

// Green, LSB+5
#define LWE2B0_ISP_SOLARIZE_CONTROL1_0_SOL_G5                        27:26

// Green, LSB+6
#define LWE2B0_ISP_SOLARIZE_CONTROL1_0_SOL_G6                        29:28

// Green, LSB+7
#define LWE2B0_ISP_SOLARIZE_CONTROL1_0_SOL_G7                        31:30


#define LWE2B0_ISP_SOLARIZE_CONTROL2_0                     (0xfd)
// Blue, LSB
#define LWE2B0_ISP_SOLARIZE_CONTROL2_0_SOL_B0                        1:0

// Blue, LSB+1
#define LWE2B0_ISP_SOLARIZE_CONTROL2_0_SOL_B1                        3:2

// Blue, LSB+2
#define LWE2B0_ISP_SOLARIZE_CONTROL2_0_SOL_B2                        5:4

// Blue, LSB+3
#define LWE2B0_ISP_SOLARIZE_CONTROL2_0_SOL_B3                        7:6

// Blue, LSB+4
#define LWE2B0_ISP_SOLARIZE_CONTROL2_0_SOL_B4                        9:8

// Blue, LSB+5
#define LWE2B0_ISP_SOLARIZE_CONTROL2_0_SOL_B5                        11:10

// Blue, LSB+6
#define LWE2B0_ISP_SOLARIZE_CONTROL2_0_SOL_B6                        13:12

// Blue, LSB+7
#define LWE2B0_ISP_SOLARIZE_CONTROL2_0_SOL_B7                        15:14

// 1 to enable solarization
#define LWE2B0_ISP_SOLARIZE_CONTROL2_0_SOL_ENABLE                    31:31

// Embosss effect
//

#define LWE2B0_ISP_EMBOSS_CONTROL_0                        (0xfe)
// 1 to enable emboss effect
#define LWE2B0_ISP_EMBOSS_CONTROL_0_EMBOSS_ENABLE                    0:0

// Polarity of the effect
#define LWE2B0_ISP_EMBOSS_CONTROL_0_EB_POS                   1:1

// Edge width in [0,7]
#define LWE2B0_ISP_EMBOSS_CONTROL_0_EB_HP_OFFSET                     4:2

// Emboss depth, U2.1 in [0.0,3.5] by 0.5 step
#define LWE2B0_ISP_EMBOSS_CONTROL_0_EB_GAIN                  7:5

// Background gray level, U8.0 in [0, 255]
#define LWE2B0_ISP_EMBOSS_CONTROL_0_EB_BG_LEVEL                      15:8

// Edge low-limit (coring), U8.0 in [0, 255]
#define LWE2B0_ISP_EMBOSS_CONTROL_0_EB_LOWLIMIT                      23:16

// Edge high-limit (clipping), U8.0 in [0, 255]
#define LWE2B0_ISP_EMBOSS_CONTROL_0_EB_HIGHLIMIT                     31:24

// Statistic gathering for Auto White Balance and Auto Exposure control
// ISP callwlates two types of statistic gathering information to assist Host/AVP in performing
// dynamic Auto White Balance (AWB) and Auto Exposure (AE):
// 1. 8-bin Histogram for each color channel in sample region is defined in M2 measuring window.
//    The histograms are stored in status registers.
//
// 2. Multi-window color average and number of saturated pixels:  MxN sets of {R, G, B, S} values;
//    where M, N is any number between 1 to 64 based on M3_H_EXT_WNUM and M3_V_EXT_WNUM registers respectively; 
//    R, G, B  are averaged pixel values callwlated within M3 measuring window; S is the number of saturated pixels 
//    within M3 measuring window based on pixel value and low/high limits in ISP_SATU_M3_CONTROL register.
//
//    Each M3 sub-window has the size of {1,2,4,8,16} x {1,2,4,8,16}.
//    The resulting M3 statistics are stored in memory at starting address defined in M3_MEM_START_ADDR.
//
//    Note about number of horiztonal M3 subwindows:
//    M3 stats data are packed into one 128-bit memory write for each four M3 subwindows in the same row.  
//    If the horizontal number of M3 subwindows is not a multiple of 4, some bits of the last 128-bit write 
//    does not have meaningful statistics for sw to use.  SW is suggested to use a multiple of 4 for the horizontal 
//    number of subwindows if possible. There is no similar suggestion for number of vertical M3 subwindows.
//
// In addition, the ISP can also callwlate one type of statistic gathering information to
// assists Host/AVP in performing Auto Focus (AF):
// 3. Multi-window edge counter:  {1, 2, 3} x {1, 2, 3} sets (sub-windows) of edge counters
//    are callwlated within M4 measuring window.
//    Each M4 sub-window has programmable width and height.
//    The resulting edge count values are stored in status registers.
// M1: Sampled average and peak values for HW white balance control

// This window must be completely within
// active window area.
#define LWE2B0_ISP_H_M1_0                  (0xff)
// Pixel number where measuring window 1
//  starts.
#define LWE2B0_ISP_H_M1_0_M1_WINDOW_LINE_START                       13:0

// Number of pixels of measuring window 1
//  in a line
#define LWE2B0_ISP_H_M1_0_M1_WINDOW_LINE_WIDTH                       29:16


// This window must be completely within
// active window area.
#define LWE2B0_ISP_V_M1_0                  (0x100)
// Line number where measuring window 1
//  starts
#define LWE2B0_ISP_V_M1_0_M1_WINDOW_FRAME_START                      13:0

// Number of lines of measuring window 1
//  in a frame
#define LWE2B0_ISP_V_M1_0_M1_WINDOW_FRAME_HEIGHT                     29:16

// Histogram in M2 window
// Extended version: 8 bins of arbitrary bin boundaries
// There are 4 histograms for R, G on G/R, G on G/B, B
// There are 8 bins for each histogram
//   [HIST_BIN_B0, HIST_BIN_B1], [HIST_BIN_B1+1, HIST_BIN_B2],
//   [HIST_BIN_B2+1, HIST_BIN_B3], [HIST_BIN_B3+1, HIST_BIN_B4],
//   [HIST_BIN_B4+1, HIST_BIN_B5], [HIST_BIN_B5+1, HIST_BIN_B6],
//   [HIST_BIN_B6+1, HIST_BIN_B7], [HIST_BIN_B7+1, HIST_BIN_B8]
// 8 bins are placed to the range [HIST_BIB_B0, HIST_BIN_B8].
// HIST_BIN_Bx, x in {1,2,3,4,5,6,7} specify bin boundaries.
// HIST_BIB_Bx < HIST_BIB_By, if x, y in {0,1,2,3,4,5,6,7,8} and x < y.  

// This window must be completely within
// active window area.
#define LWE2B0_ISP_H_M2_0                  (0x101)
// Pixel number where measuring window 2
//  starts
#define LWE2B0_ISP_H_M2_0_M2_WINDOW_LINE_START                       13:0

// Number of pixels of measuring window 2
//  in a line
#define LWE2B0_ISP_H_M2_0_M2_WINDOW_LINE_WIDTH                       29:16


// This window must be completely within
// active window area.
#define LWE2B0_ISP_V_M2_0                  (0x102)
// Line number where measuring window 2
//  starts
#define LWE2B0_ISP_V_M2_0_M2_WINDOW_FRAME_START                      13:0

// Number of lines of measuring window 2
//  in a frame
#define LWE2B0_ISP_V_M2_0_M2_WINDOW_FRAME_HEIGHT                     29:16


#define LWE2B0_ISP_M2_BIN_CONTROL_E1_0                     (0x103)
// lower limit of 1st bin
#define LWE2B0_ISP_M2_BIN_CONTROL_E1_0_HIST_BIN_B0                   9:0

// upper limit of 1st bin
#define LWE2B0_ISP_M2_BIN_CONTROL_E1_0_HIST_BIN_B1                   19:10

// upper limit of 2nd bin
//    31  rw  EXT_HIST                    init=0  // 1 to enable extended 8-bin histogram
//                                                // 0 to enable 4-bin histogram
#define LWE2B0_ISP_M2_BIN_CONTROL_E1_0_HIST_BIN_B2                   29:20


#define LWE2B0_ISP_M2_BIN_CONTROL_E2_0                     (0x104)
// upper limit of 3rd bin
#define LWE2B0_ISP_M2_BIN_CONTROL_E2_0_HIST_BIN_B3                   9:0

// upper limit of 4th bin
#define LWE2B0_ISP_M2_BIN_CONTROL_E2_0_HIST_BIN_B4                   19:10

// upper limit of 5th bin
#define LWE2B0_ISP_M2_BIN_CONTROL_E2_0_HIST_BIN_B5                   29:20


#define LWE2B0_ISP_M2_BIN_CONTROL_E3_0                     (0x105)
// upper limit of 6th bin
#define LWE2B0_ISP_M2_BIN_CONTROL_E3_0_HIST_BIN_B6                   9:0

// upper limit of 7th bin
#define LWE2B0_ISP_M2_BIN_CONTROL_E3_0_HIST_BIN_B7                   19:10

// upper limit of 8th bin
#define LWE2B0_ISP_M2_BIN_CONTROL_E3_0_HIST_BIN_B8                   29:20

// M3 windows

// This window must be completely within
// active window area.
#define LWE2B0_ISP_H_M3_0                  (0x106)
// Pixel number where measuring window 3
//  starts
#define LWE2B0_ISP_H_M3_0_M3_WINDOW_LINE_START                       13:0

// Number of pixels of measuring window 3
//  in a line
#define LWE2B0_ISP_H_M3_0_M3_WINDOW_LINE_WIDTH                       29:16


// This window must be completely within
// active window area.
#define LWE2B0_ISP_V_M3_0                  (0x107)
// Line number where measuring window 3
//  starts
#define LWE2B0_ISP_V_M3_0_M3_WINDOW_FRAME_START                      13:0

// Number of lines of measuring window 3
//  in a frame
#define LWE2B0_ISP_V_M3_0_M3_WINDOW_FRAME_HEIGHT                     29:16


#define LWE2B0_ISP_H_M3_CONTROL_E1_0                       (0x108)
// Horizontal interval of M3 sub-windows
#define LWE2B0_ISP_H_M3_CONTROL_E1_0_M3_H_INTERVAL                   13:0

// Horizontal number of M3 sub-windows
//  from 1 to 64
#define LWE2B0_ISP_H_M3_CONTROL_E1_0_M3_H_EXT_WNUM                   22:16

// Width of M3 sub-window (pixels of a kind)  0 =  1 (actual width of 2)
//  1 =  2 (actual width of 4)
//  2 =  4 (actual width of 8)
//  3 =  8 (actual width of 16)
//  4 = 16 (actual width of 32)
#define LWE2B0_ISP_H_M3_CONTROL_E1_0_M3_H_WIDTH                      26:24
#define LWE2B0_ISP_H_M3_CONTROL_E1_0_M3_H_WIDTH_HH1                        (0)
#define LWE2B0_ISP_H_M3_CONTROL_E1_0_M3_H_WIDTH_HH2                        (1)
#define LWE2B0_ISP_H_M3_CONTROL_E1_0_M3_H_WIDTH_HH4                        (2)
#define LWE2B0_ISP_H_M3_CONTROL_E1_0_M3_H_WIDTH_HH8                        (3)
#define LWE2B0_ISP_H_M3_CONTROL_E1_0_M3_H_WIDTH_HH16                       (4)

// Restriction of memory write access
// 0 for anytime when stats data is ready
// 1 for write access only in H-blanking periods
//    31 rw M3_EXTENDED                   init=0  // M3 window selection mode
//                                                // 0 for fixed selection mode
//                                                // 1 for extended mode                          
#define LWE2B0_ISP_H_M3_CONTROL_E1_0_M3_WRITE_IN_HBLANK                      30:30


#define LWE2B0_ISP_V_M3_CONTROL_E1_0                       (0x109)
// Vertical interval of M3 sub-windows
#define LWE2B0_ISP_V_M3_CONTROL_E1_0_M3_V_INTERVAL                   13:0

// Vertical number of M3 sub-windows
//  from 1 to 64
#define LWE2B0_ISP_V_M3_CONTROL_E1_0_M3_V_EXT_WNUM                   22:16

// Height of M3 sub-window (pixels of a kind)  0 =  1 (actual height of 2)
//  1 =  2 (actual height of 4)
//  2 =  4 (actual height of 8)
//  3 =  8 (actual height of 16)
//  4 = 16 (actual height of 32)
#define LWE2B0_ISP_V_M3_CONTROL_E1_0_M3_V_HEIGHT                     26:24
#define LWE2B0_ISP_V_M3_CONTROL_E1_0_M3_V_HEIGHT_VH1                       (0)
#define LWE2B0_ISP_V_M3_CONTROL_E1_0_M3_V_HEIGHT_VH2                       (1)
#define LWE2B0_ISP_V_M3_CONTROL_E1_0_M3_V_HEIGHT_VH4                       (2)
#define LWE2B0_ISP_V_M3_CONTROL_E1_0_M3_V_HEIGHT_VH8                       (3)
#define LWE2B0_ISP_V_M3_CONTROL_E1_0_M3_V_HEIGHT_VH16                      (4)

// New register for M3 stats

#define LWE2B0_ISP_SATU_M3_CONTROL_0                       (0x10a)
// lower limit of input pixel value
#define LWE2B0_ISP_SATU_M3_CONTROL_0_M3_LOW_LIMIT                    9:0

// upper limit of input pixel value
#define LWE2B0_ISP_SATU_M3_CONTROL_0_M3_HIGH_LIMIT                   25:16


#define LWE2B0_M3_MEM_START_ADDR_0                 (0x10b)
// Memory start address for storing M3
// statistics data
#define LWE2B0_M3_MEM_START_ADDR_0_M3_MEM_START_ADDR                 31:0

// M4: Statistics measurement window 4 for auto-focus control
// There will be {1, 2, 3, 4, 6, 9} sub-windows in {1,2,3} x {1,2,3} configurations.
// Top-left corner of the top-left sub-window is specified 
//  by {M4_WINDOW_FRAME_START, M4_WINDOW_LINE_START}.
// M4_V_WNUM and M4_H_WNUM independently specify the number of sub-windows in horizontal and
//  vertical directions. Selection is limited in {1,2,3} and value 0 in interpreted as 1.
// M4_V_INTERVAL and M4_H_INTERVAL specify vertical and horizontal intervals of sub-windows.
// M4_V_HEIGHT and M4_H_WIDTH specify the size of a sub-window.
// Constraints are:
//   M4_V_INTERVAL >= M4_V_HEIGHT
//   M4_H_INTERVAL >= M4_H_WIDTH 
//  Be sure to fit from top-left corner of the top-left sub-window to bottom-right corner of the
//  bottom-right sub window in the active image frame specified by
//  {ACTIVE_FRAME_START, ACTIVE_LINE_START} and {ACTIVE_FRAME_HEIGHT, ACTIVE_LINE_WIDTH}.

// This window must be completely within
// active window area.
#define LWE2B0_ISP_H_M4_0                  (0x10c)
// Pixel number where measuring window 4
//  starts
#define LWE2B0_ISP_H_M4_0_M4_WINDOW_LINE_START                       13:0


// This window must be completely within
// active window area.
#define LWE2B0_ISP_V_M4_0                  (0x10d)
// Line number where measuring window 4
//  starts
#define LWE2B0_ISP_V_M4_0_M4_WINDOW_FRAME_START                      13:0


#define LWE2B0_ISP_H_M4_CONTROL_0                  (0x10e)
// Horizontal interval of M4 sub-windows
#define LWE2B0_ISP_H_M4_CONTROL_0_M4_H_INTERVAL                      13:0

// Horizontal number of M4 sub-windows
//  0, 1 = 1 sub-window  per row
//  2    = 2 sub-windows per row
//  3    = 3 sub-windows per row
#define LWE2B0_ISP_H_M4_CONTROL_0_M4_H_WNUM                  15:14

// Number of pixels of M4 measuring sub-window
#define LWE2B0_ISP_H_M4_CONTROL_0_M4_H_WIDTH                 29:16


#define LWE2B0_ISP_V_M4_CONTROL_0                  (0x10f)
// Vertical interval of M4 sub-windows
#define LWE2B0_ISP_V_M4_CONTROL_0_M4_V_INTERVAL                      13:0

// Vertical number of M4 sub-windows
//  0, 1 = 1 sub-window  per column
//  2    = 2 sub-windows per column
//  3    = 3 sub-windows per column
#define LWE2B0_ISP_V_M4_CONTROL_0_M4_V_WNUM                  15:14

// Number of lines of M4 measuring sub-window
#define LWE2B0_ISP_V_M4_CONTROL_0_M4_V_HEIGHT                        29:16


#define LWE2B0_ISP_M4_NOISE_CONTROL_0                      (0x110)
// Top limiting level of noise coring for 
// auto-focus measuring signal
#define LWE2B0_ISP_M4_NOISE_CONTROL_0_AFDCORELIMIT                   5:0

// Flicker-Band detection
//   Statistics data (column vector) acquisition and flicker band frequency estimation
//   Note that FB is operational only when special effects are off (emboss, negative, solarize)
// 
// FB control registers

#define LWE2B0_ISP_FB_STATS_CONTROL1_0                     (0x111)
// Continuously gather FB stats  0 = disable
//  1 = enable
//  when enabled FB_CONTROL will continuously
//  gather flicker band stats, with the frame interval 
//  specified by FB_STATS_FRAME_GAP.
#define LWE2B0_ISP_FB_STATS_CONTROL1_0_CONTINUOUS_DETECTION                  0:0
#define LWE2B0_ISP_FB_STATS_CONTROL1_0_CONTINUOUS_DETECTION_DISABLE                        (0)
#define LWE2B0_ISP_FB_STATS_CONTROL1_0_CONTINUOUS_DETECTION_ENABLE                 (1)

// Start scan-line for FB stats sampling; Note that
//  if FB sampling begins before Active Window starts,
//  the sampled stats will be zero, since FB stats 
//  samples the output pixel values from demosaic output
#define LWE2B0_ISP_FB_STATS_CONTROL1_0_FB_STATS_FRAME_START                  14:1

// Vertical sampling interval (lines)
//  0 means every line is sampled, 1 means every other line,
//  and so on.
#define LWE2B0_ISP_FB_STATS_CONTROL1_0_FB_STATS_SAMPLE_INTERVAL                      19:16

// Number of lines aclwmulated per interval 0 = 1 line
// 1 = 2 lines
// 2 = 4 lines
// 3 = 8 lines
// 4 = 16 lines
//  If number of aclwmulating lines indicated here is
//  larger than FB_STATS_SAMPLE_INTERVAL, only lines 
//  sampling interval will be aclwmulated.
#define LWE2B0_ISP_FB_STATS_CONTROL1_0_FB_STATS_LINES_PER_SAMPLE                     22:20
#define LWE2B0_ISP_FB_STATS_CONTROL1_0_FB_STATS_LINES_PER_SAMPLE_LINE1                     (0)
#define LWE2B0_ISP_FB_STATS_CONTROL1_0_FB_STATS_LINES_PER_SAMPLE_LINE2                     (1)
#define LWE2B0_ISP_FB_STATS_CONTROL1_0_FB_STATS_LINES_PER_SAMPLE_LINE4                     (2)
#define LWE2B0_ISP_FB_STATS_CONTROL1_0_FB_STATS_LINES_PER_SAMPLE_LINE8                     (3)
#define LWE2B0_ISP_FB_STATS_CONTROL1_0_FB_STATS_LINES_PER_SAMPLE_LINE16                    (4)

// Temporal gap between 2 frames (frames)
#define LWE2B0_ISP_FB_STATS_CONTROL1_0_FB_STATS_FRAME_GAP                    26:23


#define LWE2B0_ISP_FB_STATS_CONTROL2_0                     (0x112)
// Starting pixel for FB stats sampling in a line
#define LWE2B0_ISP_FB_STATS_CONTROL2_0_FB_STATS_LINE_START                   13:0

// Number of sampling pixels per line
#define LWE2B0_ISP_FB_STATS_CONTROL2_0_FB_STATS_LINE_WIDTH                   29:16

// Enable normalization of line aclwmulation value;
//  if disabled the line aclwmulation value will be
//  clipped to 16-bit boundary. 
//  0    Disable
//  1    Enable
#define LWE2B0_ISP_FB_STATS_CONTROL2_0_FB_STATS_VNORM_ENABLE                 31:31


#define LWE2B0_ISP_FB_STATS_CONTROL3_0                     (0x113)
// Length of a column vector minus 1. 
//  0 means the FB stats column vector has only one entry,
//  0xff means the column vector has 256 entries.
#define LWE2B0_ISP_FB_STATS_CONTROL3_0_FB_STATS_VECTOR_LENGTH                        7:0


#define LWE2B0_ISP_FB_STATS_CONTROL4_0                     (0x114)
// Memory start address for storing FB
// statistics data
#define LWE2B0_ISP_FB_STATS_CONTROL4_0_FB_MEM_START_ADDR                     31:0


#define LWE2B0_ISP_FB_STATS_CONTROL5_0                     (0x115)
// U0.W, Filter coefficient for column mean
//  callwlation.
#define LWE2B0_ISP_FB_STATS_CONTROL5_0_FB_MEAN_FLTR_COEF                     3:0

// U0.W, Coefficient of column vector
//  smoothing filter.
#define LWE2B0_ISP_FB_STATS_CONTROL5_0_FB_DATA_FLTR_COEF                     7:4

// U0.W, Coefficient of temporal smoothing
//  filter for FB scores.
#define LWE2B0_ISP_FB_STATS_CONTROL5_0_FB_SCORE_FLTR_COEF                    11:8

// Number of entries in column vector
//  that is equivalent of 60Hz.
#define LWE2B0_ISP_FB_STATS_CONTROL5_0_FB_60HZ_INTERVAL                      23:16

// Number of entries in column vector
//  that is equivalent of 50Hz.
#define LWE2B0_ISP_FB_STATS_CONTROL5_0_FB_50HZ_INTERVAL                      31:24


//  Higher score means higher possibility that
//  such flicker band appears on image
#define LWE2B0_STATS_FB_DETECTION_SCORES_0                 (0x116)
// score for 60Hz estimation
#define LWE2B0_STATS_FB_DETECTION_SCORES_0_FB_60HZ_SCORE                     7:0

// score for 50Hz estimation
#define LWE2B0_STATS_FB_DETECTION_SCORES_0_FB_50HZ_SCORE                     23:16

// STATS command register

// When this command is issued it will take
//  effect in the next frame. This command
//  may be used to enable M2, M3, or M4
//  statistic gathering. If more than one type
//  of statistic gathering is enabled then
//  all the enabled statistic gathering will
//  be done in the next active window area.
//  Raise vector (if raise is enabled) and
//  interrupt (if enabled) are generated in
//  the same frame where statistic gathering
//  is performed at the point where line counter
//  reaches STATS_LINE_END. Raise vector or
//  interrupt may still be generated in similar
//  if this command is issued with none of the
//  statistic gathering is enabled.
#define LWE2B0_ISP_STATS_COMMAND_0                 (0x117)
// Statistic Gathering raise  0 = disable
//  1 = enable - raise vector will be returned
//      when vertical counter reaches
//      STATS_LINE_END in the frame where
//      statistic gathering is performed.
#define LWE2B0_ISP_STATS_COMMAND_0_STATS_RAISE                       0:0
#define LWE2B0_ISP_STATS_COMMAND_0_STATS_RAISE_DISABLE                     (0)
#define LWE2B0_ISP_STATS_COMMAND_0_STATS_RAISE_ENABLE                      (1)

// M2 Statistic Gathering enable  0 = disable M2 statistic gathering
//  1 = enable M2 statistic gathering
#define LWE2B0_ISP_STATS_COMMAND_0_M2_STATS_ENABLE                   1:1
#define LWE2B0_ISP_STATS_COMMAND_0_M2_STATS_ENABLE_DISABLE                 (0)
#define LWE2B0_ISP_STATS_COMMAND_0_M2_STATS_ENABLE_ENABLE                  (1)

// M3 Statistic Gathering enable  0 = disable M3 statistic gathering
//  1 = enable M3 statistic gathering
#define LWE2B0_ISP_STATS_COMMAND_0_M3_STATS_ENABLE                   2:2
#define LWE2B0_ISP_STATS_COMMAND_0_M3_STATS_ENABLE_DISABLE                 (0)
#define LWE2B0_ISP_STATS_COMMAND_0_M3_STATS_ENABLE_ENABLE                  (1)

// M4 Statistic Gathering enable  0 = disable M4 statistic gathering
//  1 = enable M4 statistic gathering
#define LWE2B0_ISP_STATS_COMMAND_0_M4_STATS_ENABLE                   3:3
#define LWE2B0_ISP_STATS_COMMAND_0_M4_STATS_ENABLE_DISABLE                 (0)
#define LWE2B0_ISP_STATS_COMMAND_0_M4_STATS_ENABLE_ENABLE                  (1)

// Channel that issues this Raise
// This channel ID is returned when the
// programmed signal raise event oclwrred.
#define LWE2B0_ISP_STATS_COMMAND_0_STATS_CHANNEL_ID                  7:4

// Statistic Gathering Raise Vector
// This raise vector is returned when
//  the last M3 measuring window after all
//  M3 statistic gathering is completed.
#define LWE2B0_ISP_STATS_COMMAND_0_STATS_RAISE_VECTOR                        12:8

// Line number to return the raise vector or
//  to generate interrupt in the frame where
//  statistic gathering is performed
#define LWE2B0_ISP_STATS_COMMAND_0_STATS_LINE_END                    29:16

// FB command register
// In AP15, no flicker band "detection", only flicker band statistic gathering

// The command takes effect at the nearest vsync.
//  If CONTINUOUS_DETECTION is "enabled", 
//  the column vectors from two frames will 
//  be callwlated and stored to memory continuously with the interval
//  specified in FB_STATS_SAMPLE_INTERVAL.  When 
//  FB stat gathering command is issued, the raise 
//  vector will be returned right after the completion
//  of the nearest FB stats callwlation.  
//  If CONTINUOUS_DETECTION is off, when the FB command
//  is issued the column vectors of the two frames will 
//  still be stored to memory, but only once.  The raise will 
//  be returned when the FB stats callwlation is completed
//  (the last sampling line is FB_STATS_FRAME_START + 
//  FB_STATS_SAMPLE_INTERVAL * 2^FB_STATS_LINES_PER_SAMPLE)
#define LWE2B0_ISP_FB_STATS_COMMAND_0                      (0x118)
// FB Statistic Gathering raise  0 = disable
//  1 = enable 
#define LWE2B0_ISP_FB_STATS_COMMAND_0_FB_STATS_RAISE                 0:0
#define LWE2B0_ISP_FB_STATS_COMMAND_0_FB_STATS_RAISE_DISABLE                       (0)
#define LWE2B0_ISP_FB_STATS_COMMAND_0_FB_STATS_RAISE_ENABLE                        (1)

// Channel that issues this Raise
// This channel ID is returned when the
// programmed signal raise event oclwrred.
#define LWE2B0_ISP_FB_STATS_COMMAND_0_FB_STATS_CHANNEL_ID                    7:4

// FB Statistic Gathering Raise Vector
#define LWE2B0_ISP_FB_STATS_COMMAND_0_FB_STATS_RAISE_VECTOR                  12:8

//  Status registers
// Operation status
//   0. Erroneous configuration for M1 windows
//   1. Erroneous configuration for M3 windows
//   2. 64x64 window samplig is done
//   3. Flicker-band stats data acquisition is done
//   4. Line buffer got read error

#define LWE2B0_OPERATION_STATUS_0                  (0x119)
// M1 window configuration error 
#define LWE2B0_OPERATION_STATUS_0_M1_CONFIG_ERROR                    0:0

// M3 window configuration error 
#define LWE2B0_OPERATION_STATUS_0_M3_CONFIG_ERROR                    1:1

// M3 frame sampling is completed
#define LWE2B0_OPERATION_STATUS_0_M3_SAMPLING_DONE                   2:2

// FB stats acquisition is completed
#define LWE2B0_OPERATION_STATUS_0_FB_STATS_DONE                      3:3

// Line buffer, read error
#define LWE2B0_OPERATION_STATUS_0_READERROR                  4:4

// Statistics measured values
//   1. Peak values,  {R, B, Gr, Gb}
//   2. Sampled average values,  {R, B, Gr, Gb}
//   3. Simple histogram,  4 bins for {R, B, Gr, Gb}
//   4. Sensing signal for auto focus control in 3x3 windows
//   measurement 1 and 2 are from M1 window, they are constantly updated at 
//   frame end (VSCAN_SIZE).  Software can read the registers after frame end 
//   and use the information to perform functions like scene change detection.

#define LWE2B0_STATS_PEAK_VALUE1_0                 (0x11a)
// Peak value of Red
#define LWE2B0_STATS_PEAK_VALUE1_0_SPEAKR                    9:0

// Peak value of Blue
#define LWE2B0_STATS_PEAK_VALUE1_0_SPEAKB                    25:16


#define LWE2B0_STATS_PEAK_VALUE2_0                 (0x11b)
// Peak value of Green on G/R
#define LWE2B0_STATS_PEAK_VALUE2_0_SPEAKGR                   9:0

// Peak value of Green on G/B
#define LWE2B0_STATS_PEAK_VALUE2_0_SPEAKGB                   25:16


#define LWE2B0_STATS_AVERAGE_VALUE1_0                      (0x11c)
// Average value of Red
#define LWE2B0_STATS_AVERAGE_VALUE1_0_SRAVE                  9:0

// Average value of Blue
#define LWE2B0_STATS_AVERAGE_VALUE1_0_SBAVE                  25:16


#define LWE2B0_STATS_AVERAGE_VALUE2_0                      (0x11d)
// Average value of Green on G/R
#define LWE2B0_STATS_AVERAGE_VALUE2_0_SGRAVE                 9:0

// Average value of Green on G/B
#define LWE2B0_STATS_AVERAGE_VALUE2_0_SGBAVE                 25:16


#define LWE2B0_STATS_HIST_RED_VALUE1_0                     (0x11e)
// Count value of #0 bin of Red
#define LWE2B0_STATS_HIST_RED_VALUE1_0_SHIST0R                       22:0


#define LWE2B0_STATS_HIST_RED_VALUE2_0                     (0x11f)
// Count value of #1 bin of Red
#define LWE2B0_STATS_HIST_RED_VALUE2_0_SHIST1R                       22:0


#define LWE2B0_STATS_HIST_RED_VALUE3_0                     (0x120)
// Count value of #2 bin of Red
#define LWE2B0_STATS_HIST_RED_VALUE3_0_SHIST2R                       22:0


#define LWE2B0_STATS_HIST_RED_VALUE4_0                     (0x121)
// Count value of #3 bin of Red
#define LWE2B0_STATS_HIST_RED_VALUE4_0_SHIST3R                       22:0


#define LWE2B0_STATS_HIST_RED_VALUE5_0                     (0x122)
// Count value of #4 bin of Red
#define LWE2B0_STATS_HIST_RED_VALUE5_0_SHIST4R                       22:0


#define LWE2B0_STATS_HIST_RED_VALUE6_0                     (0x123)
// Count value of #5 bin of Red
#define LWE2B0_STATS_HIST_RED_VALUE6_0_SHIST5R                       22:0


#define LWE2B0_STATS_HIST_RED_VALUE7_0                     (0x124)
// Count value of #6 bin of Red
#define LWE2B0_STATS_HIST_RED_VALUE7_0_SHIST6R                       22:0


#define LWE2B0_STATS_HIST_RED_VALUE8_0                     (0x125)
// Count value of #7 bin of Red
#define LWE2B0_STATS_HIST_RED_VALUE8_0_SHIST7R                       22:0


#define LWE2B0_STATS_HIST_GR_VALUE1_0                      (0x126)
// Count value of #0 bin of Green on G/R
#define LWE2B0_STATS_HIST_GR_VALUE1_0_SHIST0GR                       22:0


#define LWE2B0_STATS_HIST_GR_VALUE2_0                      (0x127)
// Count value of #1 bin of Green on G/R
#define LWE2B0_STATS_HIST_GR_VALUE2_0_SHIST1GR                       22:0


#define LWE2B0_STATS_HIST_GR_VALUE3_0                      (0x128)
// Count value of #2 bin of Green on G/R
#define LWE2B0_STATS_HIST_GR_VALUE3_0_SHIST2GR                       22:0


#define LWE2B0_STATS_HIST_GR_VALUE4_0                      (0x129)
// Count value of #3 bin of Green on G/R
#define LWE2B0_STATS_HIST_GR_VALUE4_0_SHIST3GR                       22:0


#define LWE2B0_STATS_HIST_GR_VALUE5_0                      (0x12a)
// Count value of #4 bin of Green on G/R
#define LWE2B0_STATS_HIST_GR_VALUE5_0_SHIST4GR                       22:0


#define LWE2B0_STATS_HIST_GR_VALUE6_0                      (0x12b)
// Count value of #5 bin of Green on G/R
#define LWE2B0_STATS_HIST_GR_VALUE6_0_SHIST5GR                       22:0


#define LWE2B0_STATS_HIST_GR_VALUE7_0                      (0x12c)
// Count value of #6 bin of Green on G/R
#define LWE2B0_STATS_HIST_GR_VALUE7_0_SHIST6GR                       22:0


#define LWE2B0_STATS_HIST_GR_VALUE8_0                      (0x12d)
// Count value of #7 bin of Green on G/R
#define LWE2B0_STATS_HIST_GR_VALUE8_0_SHIST7GR                       22:0


#define LWE2B0_STATS_HIST_GB_VALUE1_0                      (0x12e)
// Count value of #0 bin of Green on G/B
#define LWE2B0_STATS_HIST_GB_VALUE1_0_SHIST0GB                       22:0


#define LWE2B0_STATS_HIST_GB_VALUE2_0                      (0x12f)
// Count value of #1 bin of Green on G/B
#define LWE2B0_STATS_HIST_GB_VALUE2_0_SHIST1GB                       22:0


#define LWE2B0_STATS_HIST_GB_VALUE3_0                      (0x130)
// Count value of #2 bin of Green on G/B
#define LWE2B0_STATS_HIST_GB_VALUE3_0_SHIST2GB                       22:0


#define LWE2B0_STATS_HIST_GB_VALUE4_0                      (0x131)
// Count value of #3 bin of Green on G/B
#define LWE2B0_STATS_HIST_GB_VALUE4_0_SHIST3GB                       22:0


#define LWE2B0_STATS_HIST_GB_VALUE5_0                      (0x132)
// Count value of #4 bin of Green on G/B
#define LWE2B0_STATS_HIST_GB_VALUE5_0_SHIST4GB                       22:0


#define LWE2B0_STATS_HIST_GB_VALUE6_0                      (0x133)
// Count value of #5 bin of Green on G/B
#define LWE2B0_STATS_HIST_GB_VALUE6_0_SHIST5GB                       22:0


#define LWE2B0_STATS_HIST_GB_VALUE7_0                      (0x134)
// Count value of #6 bin of Green on G/B
#define LWE2B0_STATS_HIST_GB_VALUE7_0_SHIST6GB                       22:0


#define LWE2B0_STATS_HIST_GB_VALUE8_0                      (0x135)
// Count value of #7 bin of Green on G/B
#define LWE2B0_STATS_HIST_GB_VALUE8_0_SHIST7GB                       22:0


#define LWE2B0_STATS_HIST_BLUE_VALUE1_0                    (0x136)
// Count value of #0 bin of BLUE
#define LWE2B0_STATS_HIST_BLUE_VALUE1_0_SHIST0B                      22:0


#define LWE2B0_STATS_HIST_BLUE_VALUE2_0                    (0x137)
// Count value of #1 bin of BLUE
#define LWE2B0_STATS_HIST_BLUE_VALUE2_0_SHIST1B                      22:0


#define LWE2B0_STATS_HIST_BLUE_VALUE3_0                    (0x138)
// Count value of #2 bin of BLUE
#define LWE2B0_STATS_HIST_BLUE_VALUE3_0_SHIST2B                      22:0


#define LWE2B0_STATS_HIST_BLUE_VALUE4_0                    (0x139)
// Count value of #3 bin of BLUE
#define LWE2B0_STATS_HIST_BLUE_VALUE4_0_SHIST3B                      22:0


#define LWE2B0_STATS_HIST_BLUE_VALUE5_0                    (0x13a)
// Count value of #4 bin of BLUE
#define LWE2B0_STATS_HIST_BLUE_VALUE5_0_SHIST4B                      22:0


#define LWE2B0_STATS_HIST_BLUE_VALUE6_0                    (0x13b)
// Count value of #5 bin of BLUE
#define LWE2B0_STATS_HIST_BLUE_VALUE6_0_SHIST5B                      22:0


#define LWE2B0_STATS_HIST_BLUE_VALUE7_0                    (0x13c)
// Count value of #6 bin of BLUE
#define LWE2B0_STATS_HIST_BLUE_VALUE7_0_SHIST6B                      22:0


#define LWE2B0_STATS_HIST_BLUE_VALUE8_0                    (0x13d)
// Count value of #7 bin of BLUE
#define LWE2B0_STATS_HIST_BLUE_VALUE8_0_SHIST7B                      22:0


#define LWE2B0_STATS_AUTO_FOLWS_VALUE1_0                   (0x13e)
// Value from window-11
#define LWE2B0_STATS_AUTO_FOLWS_VALUE1_0_AF11D                       31:0


#define LWE2B0_STATS_AUTO_FOLWS_VALUE2_0                   (0x13f)
// Value from window-12
#define LWE2B0_STATS_AUTO_FOLWS_VALUE2_0_AF12D                       31:0


#define LWE2B0_STATS_AUTO_FOLWS_VALUE3_0                   (0x140)
// Value from window-13
#define LWE2B0_STATS_AUTO_FOLWS_VALUE3_0_AF13D                       31:0


#define LWE2B0_STATS_AUTO_FOLWS_VALUE4_0                   (0x141)
// Value from window-21
#define LWE2B0_STATS_AUTO_FOLWS_VALUE4_0_AF21D                       31:0


#define LWE2B0_STATS_AUTO_FOLWS_VALUE5_0                   (0x142)
// Value from window-22
#define LWE2B0_STATS_AUTO_FOLWS_VALUE5_0_AF22D                       31:0


#define LWE2B0_STATS_AUTO_FOLWS_VALUE6_0                   (0x143)
// Value from window-23
#define LWE2B0_STATS_AUTO_FOLWS_VALUE6_0_AF23D                       31:0


#define LWE2B0_STATS_AUTO_FOLWS_VALUE7_0                   (0x144)
// Value from window-31
#define LWE2B0_STATS_AUTO_FOLWS_VALUE7_0_AF31D                       31:0


#define LWE2B0_STATS_AUTO_FOLWS_VALUE8_0                   (0x145)
// Value from window-32
#define LWE2B0_STATS_AUTO_FOLWS_VALUE8_0_AF32D                       31:0


#define LWE2B0_STATS_AUTO_FOLWS_VALUE9_0                   (0x146)
// Value from window-33
#define LWE2B0_STATS_AUTO_FOLWS_VALUE9_0_AF33D                       31:0


#define LWE2B0_STATS_AUTO_FOLWS_LPF_VALUE1_0                       (0x147)
// Value from window-11
#define LWE2B0_STATS_AUTO_FOLWS_LPF_VALUE1_0_AFLPF11D                        31:0


#define LWE2B0_STATS_AUTO_FOLWS_LPF_VALUE2_0                       (0x148)
// Value from window-12
#define LWE2B0_STATS_AUTO_FOLWS_LPF_VALUE2_0_AFLPF12D                        31:0


#define LWE2B0_STATS_AUTO_FOLWS_LPF_VALUE3_0                       (0x149)
// Value from window-13
#define LWE2B0_STATS_AUTO_FOLWS_LPF_VALUE3_0_AFLPF13D                        31:0


#define LWE2B0_STATS_AUTO_FOLWS_LPF_VALUE4_0                       (0x14a)
// Value from window-21
#define LWE2B0_STATS_AUTO_FOLWS_LPF_VALUE4_0_AFLPF21D                        31:0


#define LWE2B0_STATS_AUTO_FOLWS_LPF_VALUE5_0                       (0x14b)
// Value from window-22
#define LWE2B0_STATS_AUTO_FOLWS_LPF_VALUE5_0_AFLPF22D                        31:0


#define LWE2B0_STATS_AUTO_FOLWS_LPF_VALUE6_0                       (0x14c)
// Value from window-23
#define LWE2B0_STATS_AUTO_FOLWS_LPF_VALUE6_0_AFLPF23D                        31:0


#define LWE2B0_STATS_AUTO_FOLWS_LPF_VALUE7_0                       (0x14d)
// Value from window-31
#define LWE2B0_STATS_AUTO_FOLWS_LPF_VALUE7_0_AFLPF31D                        31:0


#define LWE2B0_STATS_AUTO_FOLWS_LPF_VALUE8_0                       (0x14e)
// Value from window-32
#define LWE2B0_STATS_AUTO_FOLWS_LPF_VALUE8_0_AFLPF32D                        31:0


#define LWE2B0_STATS_AUTO_FOLWS_LPF_VALUE9_0                       (0x14f)
// Value from window-33
#define LWE2B0_STATS_AUTO_FOLWS_LPF_VALUE9_0_AFLPF33D                        31:0


#define LWE2B0_STATS_AUTO_FOLWS_DIF_VALUE1_0                       (0x150)
// Value from window-11
#define LWE2B0_STATS_AUTO_FOLWS_DIF_VALUE1_0_AFDIF11D                        31:0


#define LWE2B0_STATS_AUTO_FOLWS_DIF_VALUE2_0                       (0x151)
// Value from window-12
#define LWE2B0_STATS_AUTO_FOLWS_DIF_VALUE2_0_AFDIF12D                        31:0


#define LWE2B0_STATS_AUTO_FOLWS_DIF_VALUE3_0                       (0x152)
// Value from window-13
#define LWE2B0_STATS_AUTO_FOLWS_DIF_VALUE3_0_AFDIF13D                        31:0


#define LWE2B0_STATS_AUTO_FOLWS_DIF_VALUE4_0                       (0x153)
// Value from window-21
#define LWE2B0_STATS_AUTO_FOLWS_DIF_VALUE4_0_AFDIF21D                        31:0


#define LWE2B0_STATS_AUTO_FOLWS_DIF_VALUE5_0                       (0x154)
// Value from window-22
#define LWE2B0_STATS_AUTO_FOLWS_DIF_VALUE5_0_AFDIF22D                        31:0


#define LWE2B0_STATS_AUTO_FOLWS_DIF_VALUE6_0                       (0x155)
// Value from window-23
#define LWE2B0_STATS_AUTO_FOLWS_DIF_VALUE6_0_AFDIF23D                        31:0


#define LWE2B0_STATS_AUTO_FOLWS_DIF_VALUE7_0                       (0x156)
// Value from window-31
#define LWE2B0_STATS_AUTO_FOLWS_DIF_VALUE7_0_AFDIF31D                        31:0


#define LWE2B0_STATS_AUTO_FOLWS_DIF_VALUE8_0                       (0x157)
// Value from window-32
#define LWE2B0_STATS_AUTO_FOLWS_DIF_VALUE8_0_AFDIF32D                        31:0


#define LWE2B0_STATS_AUTO_FOLWS_DIF_VALUE9_0                       (0x158)
// Value from window-33
#define LWE2B0_STATS_AUTO_FOLWS_DIF_VALUE9_0_AFDIF33D                        31:0

// Legacy: May be removed
//reg ISP_M2_BIN_CONTROL                  incr1   // ISP White Balance, Control
//    (ISP_PIXEL_DIBIT-1):0   rw  HIST_BIN_L1     // upper limit of 1st bin
//    (ISP_PIXEL_DIBIT+9):10  rw  HIST_BIN_L2     // upper limit of 2nd bin
//    (ISP_PIXEL_DIBIT+19):20 rw  HIST_BIN_L3     // upper limit of 3rd bin
//;
//reg ISP_H_M3_CONTROL                    incr1   // ISP Horizontal M3 window control
//    15:14 rw  M3_H_WNUM                         // Number of M3 sub-windows in horizontal
//                                                // direction
//        enum ( HW16, HW32, HW64 )               //  0 = 16 sub-windows per row
//                                                //  1 = 32 sub-windows per row
//                                                //  2 = 64 sub-windows per row
//                                                // The vertical interval between vertically
//                                                //  adjacent windows are callwlated by dividing
//                                                //  M3_WINDOW_LINE_WIDTH by NHWINNUM and
//                                                //  rounded down to the nearest integer.
//;
// Pre-arranged selections
//reg ISP_V_M3_CONTROL                    incr1   // ISP Vertical M3 window control
//    15:14 rw  M3_V_WNUM                         // Number of M3 sub-windows in vertical
//                                                // direction
//        enum ( VW16, VW32, VW64 )               //  0 = 16 sub-windows
//                                                //  1 = 32 sub-windows
//                                                //  2 = 64 sub-windows
//                                                // The vertical interval between vertically
//                                                //  adjacent windows are callwlated by dividing
//                                                //  M3_WINDOW_FRAME_HEIGHT by LWWINNUM and
//                                                //  rounded down to the nearest integer.
//;
// Memory Client Interface Async Fifo Optimization Register
// Memory Client Interface Fifo Control Register.
// The registers below allow to optimize the synchronization timing in
// the memory client asynchronous fifos. When they can be used depend on
// the client and memory controller clock ratio.
// Additionally, the RDMC_RDFAST/RDCL_RDFAST fields can increase power
// consumption if the asynchronous fifo is implemented as a real ram.
// There is no power impact on latch-based fifos. Flipflop-based fifos
// do not use these fields.
// See recommended settings below.
//
// !! IMPORTANT !!
// The register fields can only be changed when the memory client async
// fifos are empty.
//
// The register field ending with WRCL_MCLE2X (if any) can be set to improve
// async fifo synchronization on the write side by one client clock cycle if
// the memory controller clock frequency is less or equal to twice the client
// clock frequency:
//
//      mcclk_freq <= 2 * clientclk_freq
//
// The register field ending with WRMC_CLLE2X (if any) can be set to improve
// async fifo synchronization on the write side by one memory controller clock
// cycle if the client clock frequency is less or equal to twice the memory
// controller clock frequency:
//
//      clientclk_freq <= 2 * mcclk_freq
//
// The register field ending with RDMC_RDFAST (if any) can be set to improve async
// fifo synchronization on the read side by one memory controller clock cycle.
//
// !! WARNING !!
// RDMC_RDFAST can be used along with WRCL_MCLE2X only when:
//
//       mcclk_freq <= clientclk_freq
//
// The register field ending with RDCL_RDFAST (if any) can be set to improve async
// fifo synchronization on the read side by one client clock cycle.
//
// !! WARNING !!
// RDCL_RDFAST can be used along with WRMC_CLLE2X only when:
//
//       clientclk_freq <= mcclk_freq
//
// RECOMMENDED SETTINGS
// # Client writing to fifo, memory controller reading from fifo
// - mcclk_freq <= clientclk_freq
//     You can enable both RDMC_RDFAST and WRCL_CLLE2X. If one of the fifos is
//     a real ram and power is a concern, you should avoid enabling RDMC_RDFAST.
// - clientclk_freq < mcclk_freq <= 2 * clientclk_freq
//     You can enable RDMC_RDFAST or WRCL_MCLE2X, but because the client clock
//     is slower, you should enable only WRCL_MCLE2X.
// - 2 * clientclk_freq < mcclk_freq
//     You can only enable RDMC_RDFAST. If one of the fifos is a real ram and
//     power is a concern, you should avoid enabling RDMC_RDFAST.
//
// # Memory controller writing to fifo, client reading from fifo
// - clientclk_freq <= mcclk_freq
//     You can enable both RDCL_RDFAST and WRMC_CLLE2X. If one of the fifos is
//     a real ram and power is a concern, you should avoid enabling RDCL_RDFAST.
// - mcclk_freq < clientclk_freq <= 2 * mcclk_freq
//     You can enable RDCL_RDFAST or WRMC_CLLE2X, but because the memory controller
//     clock is slower, you should enable only WRMC_CLLE2X.
// - 2 * mcclk_freq < clientclk_freq
//     You can only enable RDCL_RDFAST. If one of the fifos is a real ram and
//     power is a concern, you should avoid enabling RDCL_RDFAST.
//

#define LWE2B0_ISP_MCCIF_FIFOCTRL_0                        (0x159)
#define LWE2B0_ISP_MCCIF_FIFOCTRL_0_ISP_MCCIF_WRCL_MCLE2X                    0:0
#define LWE2B0_ISP_MCCIF_FIFOCTRL_0_ISP_MCCIF_WRCL_MCLE2X_DISABLE                  (0)
#define LWE2B0_ISP_MCCIF_FIFOCTRL_0_ISP_MCCIF_WRCL_MCLE2X_ENABLE                   (1)

#define LWE2B0_ISP_MCCIF_FIFOCTRL_0_ISP_MCCIF_RDMC_RDFAST                    1:1
#define LWE2B0_ISP_MCCIF_FIFOCTRL_0_ISP_MCCIF_RDMC_RDFAST_DISABLE                  (0)
#define LWE2B0_ISP_MCCIF_FIFOCTRL_0_ISP_MCCIF_RDMC_RDFAST_ENABLE                   (1)

// Write Coalescing Time-Out Register
// This register exists only for write clients. Reset value defaults to 
// to 50 for most clients, but may be different for certain clients.
// Write coalescing happens inside the memory client.
// Coalescing means two (LW_MC_MW/2)-bit requests are grouped together in one LW_MC_MW-bit request.
// The register value indicates how many cycles a first write request is going to wait
// for a subsequent one for possible coalescing. The coalescing can only happen
// if the request addresses are compatible. A value of zero means that coalescing is off
// and requests are sent right away to the memory controller.
// Write coalescing can have a very significant impact performance when accessing the internal memory,
// because its memory word is LW_MC_WM-bit wide. Grouping two half-word accesses is
// much more efficient, because the two accesses would actually have taken three cycles,
// due to a stall when accessing the same memory bank. It also reduces the number of
// accessing (one instead of two), freeing up internal memory bandwidth for other accesses.
// The impact on external memory accesses is not as significant as the burst access is for
// LW_MC_MW/2 bits. But a coalesced write guarantees two conselwtive same page accesses
// which is good for external memory bandwidth utilization.
// The write coalescing time-out should be programmed depending on the client behavior.
// The first write is obviously delayed by an amount of client cycles equal to the time-out value.
// Note that writes tagged by the client (i.e. the client expects a write response, usually
// for coherency), and the last write of a block transfer are not delayed.
// They only have a one-cycle opportunity to get coalesced.
//

#define LWE2B0_TIMEOUT_WCOAL_ISP_0                 (0x15a)
#define LWE2B0_TIMEOUT_WCOAL_ISP_0_ISPW_WCOAL_TMVAL                  7:0


//
// REGISTER LIST
//
#define LIST_ARISP_REGS(_op_) \
_op_(LWE2B0_INCR_SYNCPT_0) \
_op_(LWE2B0_INCR_SYNCPT_CNTRL_0) \
_op_(LWE2B0_INCR_SYNCPT_ERROR_0) \
_op_(LWE2B0_CONT_SYNCPT_FB_STATS_0) \
_op_(LWE2B0_CTXSW_0) \
_op_(LWE2B0_ISP_INT_STATUS_0) \
_op_(LWE2B0_ISP_INT_MASK_0) \
_op_(LWE2B0_ISP_INT_ENABLE_0) \
_op_(LWE2B0_ISP_SIGNAL_RAISE_0) \
_op_(LWE2B0_ISP_HP_THRESHOLD_0) \
_op_(LWE2B0_ISP_CONTROL1_0) \
_op_(LWE2B0_ISP_CONTROL2_0) \
_op_(LWE2B0_ISP_COMMAND_0) \
_op_(LWE2B0_ISP_SCAN_FRAME_0) \
_op_(LWE2B0_ISP_TG_CONTROL_0) \
_op_(LWE2B0_ISP_H_ACTIVE_0) \
_op_(LWE2B0_ISP_V_ACTIVE_0) \
_op_(LWE2B0_ISP_H_OUTPUT_0) \
_op_(LWE2B0_ISP_V_OUTPUT_0) \
_op_(LWE2B0_ISP_STALL_CONTROL_0) \
_op_(LWE2B0_ISP_OB_FCN_CONTROL1_0) \
_op_(LWE2B0_ISP_OB_FCN_CONTROL2_0) \
_op_(LWE2B0_ISP_OB_FCN_CONTROL3_0) \
_op_(LWE2B0_ISP_OB_FCN_CONTROL4_0) \
_op_(LWE2B0_ISP_OB_TOP_0) \
_op_(LWE2B0_ISP_OB_BOTTOM_0) \
_op_(LWE2B0_ISP_OB_LEFT_0) \
_op_(LWE2B0_ISP_OB_RIGHT_0) \
_op_(LWE2B0_ISP_COMMON_GAIN_CONTROL_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR1A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR1B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR2A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR2B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR3A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR3B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR4A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR4B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR5A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR5B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR6A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR6B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR7A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR7B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR8A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR8B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR9A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR9B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR10A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR10B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR11A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR11B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR12A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR12B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR13A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR13B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR14A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR14B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR15A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR15B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR16A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GR16B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB1A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB1B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB2A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB2B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB3A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB3B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB4A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB4B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB5A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB5B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB6A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB6B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB7A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB7B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB8A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB8B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB9A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB9B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB10A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB10B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB11A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB11B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB12A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB12B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB13A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB13B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB14A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB14B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB15A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB15B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB16A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_GB16B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R1A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R1B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R2A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R2B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R3A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R3B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R4A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R4B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R5A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R5B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R6A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R6B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R7A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R7B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R8A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R8B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R9A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R9B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R10A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R10B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R11A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R11B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R12A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R12B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R13A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R13B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R14A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R14B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R15A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R15B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R16A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_R16B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B1A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B1B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B2A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B2B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B3A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B3B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B4A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B4B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B5A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B5B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B6A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B6B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B7A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B7B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B8A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B8B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B9A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B9B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B10A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B10B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B11A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B11B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B12A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B12B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B13A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B13B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B14A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B14B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B15A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B15B_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B16A_0) \
_op_(LWE2B0_ISP_DEKNEE_CONFIG_B16B_0) \
_op_(LWE2B0_ISP_LS_LPATCH_WIDTH_0) \
_op_(LWE2B0_ISP_LS_CPATCH_WIDTH_0) \
_op_(LWE2B0_ISP_LS_TPATCH_HEIGHT_0) \
_op_(LWE2B0_ISP_LS_MPATCH_HEIGHT_0) \
_op_(LWE2B0_ISP_LS_LPATCH_DELTAU_MSB_0) \
_op_(LWE2B0_ISP_LS_LPATCH_DELTAU_LSB_0) \
_op_(LWE2B0_ISP_LS_CPATCH_DELTAU_MSB_0) \
_op_(LWE2B0_ISP_LS_CPATCH_DELTAU_LSB_0) \
_op_(LWE2B0_ISP_LS_RPATCH_DELTAU_MSB_0) \
_op_(LWE2B0_ISP_LS_RPATCH_DELTAU_LSB_0) \
_op_(LWE2B0_ISP_LS_TPATCH_DELTAV_MSB_0) \
_op_(LWE2B0_ISP_LS_TPATCH_DELTAV_LSB_0) \
_op_(LWE2B0_ISP_LS_MPATCH_DELTAV_MSB_0) \
_op_(LWE2B0_ISP_LS_MPATCH_DELTAV_LSB_0) \
_op_(LWE2B0_ISP_LS_BPATCH_DELTAV_MSB_0) \
_op_(LWE2B0_ISP_LS_BPATCH_DELTAV_LSB_0) \
_op_(LWE2B0_ISP_LS_CTRL_PT_OFFSET_0) \
_op_(LWE2B0_ISP_LS_CTRL_PT_DATA_0) \
_op_(LWE2B0_ISP_LS_CTRL_PT_BUFFER_0) \
_op_(LWE2B0_ISP_WBALANCE_PROG1_0) \
_op_(LWE2B0_ISP_WBALANCE_PROG2_0) \
_op_(LWE2B0_ISP_WBALANCE_CONTROL1_0) \
_op_(LWE2B0_ISP_WBALANCE_CONTROL2_0) \
_op_(LWE2B0_ISP_WBALANCE_CONTROL3_0) \
_op_(LWE2B0_ISP_WBALANCE_CONTROL4_0) \
_op_(LWE2B0_ISP_WBALANCE_CONTROL5_0) \
_op_(LWE2B0_ISP_WBALANCE_CONTROL6_0) \
_op_(LWE2B0_ISP_ENHANCE_CONFIG1_0) \
_op_(LWE2B0_ISP_ENHANCE_CONFIG2_0) \
_op_(LWE2B0_ISP_ENHANCE_CONFIG3_0) \
_op_(LWE2B0_ISP_DM_FORMULA_G11_0) \
_op_(LWE2B0_ISP_DM_FORMULA_G12_0) \
_op_(LWE2B0_ISP_DM_FORMULA_G21_0) \
_op_(LWE2B0_ISP_DM_FORMULA_G22_0) \
_op_(LWE2B0_ISP_DM_FORMULA_C11_0) \
_op_(LWE2B0_ISP_DM_FORMULA_C12_0) \
_op_(LWE2B0_ISP_DM_FORMULA_C21_0) \
_op_(LWE2B0_ISP_DM_FORMULA_C22_0) \
_op_(LWE2B0_ISP_DM_FORMULA_C31_0) \
_op_(LWE2B0_ISP_DM_FORMULA_C32_0) \
_op_(LWE2B0_ISP_DM_FORMULA_C41_0) \
_op_(LWE2B0_ISP_DM_FORMULA_C42_0) \
_op_(LWE2B0_ISP_CAR_CONTROL1_0) \
_op_(LWE2B0_ISP_CAR_CONTROL2_0) \
_op_(LWE2B0_ISP_DS_VERTICAL_0) \
_op_(LWE2B0_ISP_DS_HORIZONTAL_0) \
_op_(LWE2B0_ISP_DS_DEST_SIZE_0) \
_op_(LWE2B0_ISP_COLORCORRECT_RCONFIG1_0) \
_op_(LWE2B0_ISP_COLORCORRECT_RCONFIG2_0) \
_op_(LWE2B0_ISP_COLORCORRECT_GCONFIG1_0) \
_op_(LWE2B0_ISP_COLORCORRECT_GCONFIG2_0) \
_op_(LWE2B0_ISP_COLORCORRECT_BCONFIG1_0) \
_op_(LWE2B0_ISP_COLORCORRECT_BCONFIG2_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG1_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG2_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG3_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG4_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG5_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG6_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG7_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG8_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG9_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG10_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG11_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG12_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG13_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG14_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG15_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG16_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG17_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG18_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG19_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG20_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG21_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG22_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG23_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG24_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG25_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG26_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG27_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG28_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG29_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG30_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG31_0) \
_op_(LWE2B0_ISP_GAMMA_CONFIG32_0) \
_op_(LWE2B0_ISP_CSC_YCONFIG1_0) \
_op_(LWE2B0_ISP_CSC_YCONFIG2_0) \
_op_(LWE2B0_ISP_CSC_UCONFIG_0) \
_op_(LWE2B0_ISP_CSC_VCONFIG_0) \
_op_(LWE2B0_ISP_CSC_UVOFFSETCONFIG_0) \
_op_(LWE2B0_ISP_SOLARIZE_CONTROL1_0) \
_op_(LWE2B0_ISP_SOLARIZE_CONTROL2_0) \
_op_(LWE2B0_ISP_EMBOSS_CONTROL_0) \
_op_(LWE2B0_ISP_H_M1_0) \
_op_(LWE2B0_ISP_V_M1_0) \
_op_(LWE2B0_ISP_H_M2_0) \
_op_(LWE2B0_ISP_V_M2_0) \
_op_(LWE2B0_ISP_M2_BIN_CONTROL_E1_0) \
_op_(LWE2B0_ISP_M2_BIN_CONTROL_E2_0) \
_op_(LWE2B0_ISP_M2_BIN_CONTROL_E3_0) \
_op_(LWE2B0_ISP_H_M3_0) \
_op_(LWE2B0_ISP_V_M3_0) \
_op_(LWE2B0_ISP_H_M3_CONTROL_E1_0) \
_op_(LWE2B0_ISP_V_M3_CONTROL_E1_0) \
_op_(LWE2B0_ISP_SATU_M3_CONTROL_0) \
_op_(LWE2B0_M3_MEM_START_ADDR_0) \
_op_(LWE2B0_ISP_H_M4_0) \
_op_(LWE2B0_ISP_V_M4_0) \
_op_(LWE2B0_ISP_H_M4_CONTROL_0) \
_op_(LWE2B0_ISP_V_M4_CONTROL_0) \
_op_(LWE2B0_ISP_M4_NOISE_CONTROL_0) \
_op_(LWE2B0_ISP_FB_STATS_CONTROL1_0) \
_op_(LWE2B0_ISP_FB_STATS_CONTROL2_0) \
_op_(LWE2B0_ISP_FB_STATS_CONTROL3_0) \
_op_(LWE2B0_ISP_FB_STATS_CONTROL4_0) \
_op_(LWE2B0_ISP_FB_STATS_CONTROL5_0) \
_op_(LWE2B0_STATS_FB_DETECTION_SCORES_0) \
_op_(LWE2B0_ISP_STATS_COMMAND_0) \
_op_(LWE2B0_ISP_FB_STATS_COMMAND_0) \
_op_(LWE2B0_OPERATION_STATUS_0) \
_op_(LWE2B0_STATS_PEAK_VALUE1_0) \
_op_(LWE2B0_STATS_PEAK_VALUE2_0) \
_op_(LWE2B0_STATS_AVERAGE_VALUE1_0) \
_op_(LWE2B0_STATS_AVERAGE_VALUE2_0) \
_op_(LWE2B0_STATS_HIST_RED_VALUE1_0) \
_op_(LWE2B0_STATS_HIST_RED_VALUE2_0) \
_op_(LWE2B0_STATS_HIST_RED_VALUE3_0) \
_op_(LWE2B0_STATS_HIST_RED_VALUE4_0) \
_op_(LWE2B0_STATS_HIST_RED_VALUE5_0) \
_op_(LWE2B0_STATS_HIST_RED_VALUE6_0) \
_op_(LWE2B0_STATS_HIST_RED_VALUE7_0) \
_op_(LWE2B0_STATS_HIST_RED_VALUE8_0) \
_op_(LWE2B0_STATS_HIST_GR_VALUE1_0) \
_op_(LWE2B0_STATS_HIST_GR_VALUE2_0) \
_op_(LWE2B0_STATS_HIST_GR_VALUE3_0) \
_op_(LWE2B0_STATS_HIST_GR_VALUE4_0) \
_op_(LWE2B0_STATS_HIST_GR_VALUE5_0) \
_op_(LWE2B0_STATS_HIST_GR_VALUE6_0) \
_op_(LWE2B0_STATS_HIST_GR_VALUE7_0) \
_op_(LWE2B0_STATS_HIST_GR_VALUE8_0) \
_op_(LWE2B0_STATS_HIST_GB_VALUE1_0) \
_op_(LWE2B0_STATS_HIST_GB_VALUE2_0) \
_op_(LWE2B0_STATS_HIST_GB_VALUE3_0) \
_op_(LWE2B0_STATS_HIST_GB_VALUE4_0) \
_op_(LWE2B0_STATS_HIST_GB_VALUE5_0) \
_op_(LWE2B0_STATS_HIST_GB_VALUE6_0) \
_op_(LWE2B0_STATS_HIST_GB_VALUE7_0) \
_op_(LWE2B0_STATS_HIST_GB_VALUE8_0) \
_op_(LWE2B0_STATS_HIST_BLUE_VALUE1_0) \
_op_(LWE2B0_STATS_HIST_BLUE_VALUE2_0) \
_op_(LWE2B0_STATS_HIST_BLUE_VALUE3_0) \
_op_(LWE2B0_STATS_HIST_BLUE_VALUE4_0) \
_op_(LWE2B0_STATS_HIST_BLUE_VALUE5_0) \
_op_(LWE2B0_STATS_HIST_BLUE_VALUE6_0) \
_op_(LWE2B0_STATS_HIST_BLUE_VALUE7_0) \
_op_(LWE2B0_STATS_HIST_BLUE_VALUE8_0) \
_op_(LWE2B0_STATS_AUTO_FOLWS_VALUE1_0) \
_op_(LWE2B0_STATS_AUTO_FOLWS_VALUE2_0) \
_op_(LWE2B0_STATS_AUTO_FOLWS_VALUE3_0) \
_op_(LWE2B0_STATS_AUTO_FOLWS_VALUE4_0) \
_op_(LWE2B0_STATS_AUTO_FOLWS_VALUE5_0) \
_op_(LWE2B0_STATS_AUTO_FOLWS_VALUE6_0) \
_op_(LWE2B0_STATS_AUTO_FOLWS_VALUE7_0) \
_op_(LWE2B0_STATS_AUTO_FOLWS_VALUE8_0) \
_op_(LWE2B0_STATS_AUTO_FOLWS_VALUE9_0) \
_op_(LWE2B0_STATS_AUTO_FOLWS_LPF_VALUE1_0) \
_op_(LWE2B0_STATS_AUTO_FOLWS_LPF_VALUE2_0) \
_op_(LWE2B0_STATS_AUTO_FOLWS_LPF_VALUE3_0) \
_op_(LWE2B0_STATS_AUTO_FOLWS_LPF_VALUE4_0) \
_op_(LWE2B0_STATS_AUTO_FOLWS_LPF_VALUE5_0) \
_op_(LWE2B0_STATS_AUTO_FOLWS_LPF_VALUE6_0) \
_op_(LWE2B0_STATS_AUTO_FOLWS_LPF_VALUE7_0) \
_op_(LWE2B0_STATS_AUTO_FOLWS_LPF_VALUE8_0) \
_op_(LWE2B0_STATS_AUTO_FOLWS_LPF_VALUE9_0) \
_op_(LWE2B0_STATS_AUTO_FOLWS_DIF_VALUE1_0) \
_op_(LWE2B0_STATS_AUTO_FOLWS_DIF_VALUE2_0) \
_op_(LWE2B0_STATS_AUTO_FOLWS_DIF_VALUE3_0) \
_op_(LWE2B0_STATS_AUTO_FOLWS_DIF_VALUE4_0) \
_op_(LWE2B0_STATS_AUTO_FOLWS_DIF_VALUE5_0) \
_op_(LWE2B0_STATS_AUTO_FOLWS_DIF_VALUE6_0) \
_op_(LWE2B0_STATS_AUTO_FOLWS_DIF_VALUE7_0) \
_op_(LWE2B0_STATS_AUTO_FOLWS_DIF_VALUE8_0) \
_op_(LWE2B0_STATS_AUTO_FOLWS_DIF_VALUE9_0) \
_op_(LWE2B0_ISP_MCCIF_FIFOCTRL_0) \
_op_(LWE2B0_TIMEOUT_WCOAL_ISP_0)

//
// ADDRESS SPACES
//
#define BASE_ADDRESS_ISP        0x000000000000000

//
// ARISP REGISTER BANKS
//
#define ISP0_FIRST_REG 0x00000000000 // ISP_INCR_SYNCPT_0
#define ISP0_LAST_REG 0x00000000002 // ISP_INCR_SYNCPT_ERROR_0
#define ISP1_FIRST_REG 0x00000000008 // ISP_CONT_SYNCPT_FB_STATS_0
#define ISP1_LAST_REG 0x0000000015a // ISP_TIMEOUT_WCOAL_ISP_0

#endif // ifndef _cl_e2b0_h_
