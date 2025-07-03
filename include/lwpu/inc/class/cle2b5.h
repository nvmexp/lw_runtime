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
#ifndef _cl_e2b5_h_
#define _cl_e2b5_h_

#include "lwtypes.h"
#define LWE2_EPP                   (0xE2B5)

/*
 * Memory Controller Tiling definitions
 *
 *
 *  To enable tiling for a buffer in your module you'll want to include
 *  this spec file and then make use of either the ADD_TILE_MODE_REG_SPEC
 *  or ADD_TILE_MODE_REG_FIELD_SPEC macro.
 *
 *  For the ADD_TILE_MODE_REG_SPEC macro, the regp arg is added to the
 *  register name as a prefix to match the names of the other registers
 *  for this buffer. The fldp is the field name prefix to make the name
 *  unique so it works with arreggen generated reg blocks (e.g.):
 *
 *      specify how addressing should occur for IB0 buffer
 *      ADD_TILE_MODE_REG_SPEC(IB0, IB0);
 *
 *  There's also a REG_RW_SPEC version, if you need to specify a special
 *  flag (e.g. rws for shadow, or rwt for trigger).
 *  
 *  For the ADD_TILE_MODE_REG_FIELD_SPEC macro, the fldp is the field
 *  name prefix and bitpos arg describes the starting bit position for
 *  this field within another register.
 *
 *  Like the register version, there's a REG_RW_FIELD_SPEC version if
 *  you need to set explicit bits other than "rw".
 *
 *  Note: this requires having at least LW_MC_TILE_MODEWIDTH bits of
 *  space available after bitpos (e.g.) in the register:
 *
 *      ADD_TILE_MODE_REG_FIELD_SPEC(REF, 16)    * This parameter specifies how addressing
 *                                               * for the REF buffer should occur
 *
 * Luma/chroma address widths
 * Chroma buffer stride
 * EPP (Encoder Pre-Processor) receives input stream or surface data from VI, or 2D/SB, or
 * Display Controller, or Display Controller B and write data in 1 or more buffers in the
 * internal or external memories.
 *
 * EPP receives input data in a 24-bit data bus with format compatible to CSI output format
 * defined in  *hw/ar/doc/sc25/csi_data_format.xls. The various combination of input data
 * formats boil down to 6 possible combination as far as EPP is concern:
 * a. 24-bit YUV444 (Y in bits 23:16, U in bits 15:8, V in bits 7:0)
 * b. 24-bit RGB888 (R in bits 23:16, G in bits 15:8, B in bits 7:0)
 * c. 1-byte raw in bits 7:0
 * d. 2-byte raw in bits 15:0
 * e. MIPI CSI YUV420 8-bit legacy
 * f. MIPI CSI YUV420 8-bit
 * Note that SC17 does not support both MIPI CSI YUV420 input formats.
 *
 * EPP supports various output data formats to memory: planar YUV, non-planar YUV, RGB, and
 * raw formats.
 *
 * EPP supports the following YUV planar output formats:
 * a. YUV420P: planar YUV420 (3-plane YUV420)
 *    Y-plane:  YYYY...
 *              YYYY...
 *    U-plane:  UU...
 *    V-plane:  VV...
 * b. YUV422P: planar YUV422 (3-plane YUV422)
 *    Y-plane:  YYYY...
 *    U-plane:  UU...
 *    V-plane:  VV...
 * c. YUV422R: planar YUV422 rotated (3-plane YUV422 rotated)
 *    Y-plane:  YYYY...
 *              YYYY...
 *    U-plane:  UUUU...
 *    V-plane:  VVVV...
 * d. YUV420SP: semi-planar YUV420 (2-plane YUV420)
 *    Y-plane:  YYYY...
 *              YYYY...
 *    UV-plane: UVUV...
 * e. YUV422SP: semi-planar YUV422 (2-plane YUV422)
 *    Y-plane:  YYYY...
 *    UV-plane: UVUV...
 * f. YUV422SPR: semi-planar YUV422 rotated (2-plane YUV422 rotated)
 *    Y-plane:  YY...
 *              YY...
 *    UV-plane: UVUV...
 *
 * EPP supports the following YUV and RGB 32-bit non-planar output formats:
 *                              LSB--------------------------MSB [0:31]
 * a. YUVV444                   VVVVVVVVUUUUUUUUYYYYYYYYAAAAAAAA
 *                              01234567012345670123456701234567
 * b. YUV422NPVYUY              VVVVVVVVYYYYYYYYUUUUUUUUYYYYYYYY
 *                              00000000000000000000000011111111
 *                              01234567012345670123456701234567
 * c. YUV422NPYVYU              YYYYYYYYVVVVVVVVYYYYYYYYUUUUUUUU
 *                              00000000000000001111111100000000
 *                              01234567012345670123456701234567
 * d. YUV422NPUYVY              UUUUUUUUYYYYYYYYVVVVVVVVYYYYYYYY
 *                              00000000000000000000000011111111
 *                              01234567012345670123456701234567
 * e. YUV422NPYUYV              YYYYYYYYUUUUUUUUYYYYYYYYVVVVVVVV
 *                              11111111000000000000000000000000
 *                              01234567012345670123456701234567
 * f. B8G8R8A8                  BBBBBBBBGGGGGGGGRRRRRRRRAAAAAAAA
 *                              01234567012345670123456701234567
 * g. R8G8B8A8                  RRRRRRRRGGGGGGGGBBBBBBBBAAAAAAAA
 *                              01234567012345670123456701234567
 * h. A8B8G8R8                  AAAAAAAABBBBBBBBGGGGGGGGRRRRRRRR
 *                              01234567012345670123456701234567
 * i. A8R8G8B8                  AAAAAAAARRRRRRRRGGGGGGGGBBBBBBBB
 *                              01234567012345670123456701234567
 *
 * EPP supports the following RGB 16-bit non-planar output formats:
 *                              LSB -------- MSB [0:15]
 * a. B5G6R5                    BBBBBGGGGGGRRRRR
 *                              0123401234501234
 * b. R5G6R5                    RRRRRGGGGGGBBBBB
 *                              0123401234501234
 * c. B5G5R5A1                  BBBBBGGGGGRRRRRA
 *                              0123401234012340
 * d. R5G5B5A1                  RRRRRGGGGGBBBBBA
 *                              0123401234012340
 * e. A1B5G5R5                  ABBBBBGGGGGRRRRR
 *                              0012340123401234
 * f. A1R5G5B5                  ARRRRRGGGGGBBBBB
 *                              0012340123401234
 * g. B4G4R4A4                  BBBBGGGGRRRRAAAA
 *                              0123012301230123
 * h. R4G4B4A4                  RRRRGGGGBBBBAAAA
 *                              0123012301230123
 * i. A4B4G4R4                  AAAABBBBGGGGRRRR
 *                              0123012301230123
 * j. A4R4G4B4                  AAAARRRRGGGGBBBB
 *                              0123012301230123
 *
 * EPP supports the following non-planar raw output formats:
 *                              LSB--MSB [0:7]
 * a. RAW8 (1-byte raw)         DDDDDDDD
 *                              01234567
 *
 *                              LSB -------- MSB [0:15]
 * b. RAW16 (2-byte raw)        DDDDDDDDDDDDDDDD
 *                              0123456789111111
 *                                        012345
 *
 * For all output formats that require alpha (A), the alpha value is programmed by host and
 * the same programmed value is used for all output pixels.
 *
 * If EPP output processing is stalled due to stall from memory controllers or due to output
 * trigger stall, EPP will propagate the stall condition to its input FIFO. If input FIFO
 * becomes full, then EPP will not accept any more input data.
 * This stall may cause input data stream to be dropped (thrown away) if input stream cannot
 * be stalled or if input stream comes from host via VI, then VI will stall host input stream
 * processing.
 *
 * reset sequence for EPP added by Yuanyuan on 8-3-06. The RTL test is simpleyuv420_reset.js,
 * which is checked in at /home/aurora/traces/epp/js.
 * 1. raise VI at end of EPP frame.
 *    If the first_output of VI is connected to EPP, then RAISE_FRAME_1_VECTOR should be
 *    send to register LW_VI_RAISE_VIP_FRAME_FIRST_OUTPUT_0 (offset 0x3e). If the second
 *    output of VI is connected to EPP, then RAISE_FRAME_2_VECTOR should be send to register
 *    LW_VI_RAISE_VIP_FRAME_SECOND_OUTPUT_0 (offset 0x40).
 *2. disable VI output to EPP by set bit 4:2 to 0 of register LW_VI_VI_CORE_CONTROL_0 (offset 0x3)
 *3. raise EPP at end of frame from host, LW_EPP_RAISE_FRAME_0 (offset 0x1a)
 *4. read LW_EPP_CTXW_0 (0ffset 0) register, save the value as EPP_CTXSW_DATA
 *5. write 0 to LW_MC_CLIENT_CTRL_0 (offset 0x18) bit 4 to block EPP requests inside MC.
 *6. reset EPP by write 0 to bit 6 of LW_HOST1X_ASYNC_RSTREG_0 (offset 0x14). Do read modify write.
 *7. write 0 to LW_MC_CLIENT_HOTRESETN_0 (offset 0x19) bit 4 to clear EPP requests blocked inside MC.
 *8. poll on LW_MC_EPP_ORRC_0 (offset 0x1e), MC_EPP_ORRC_0_EPP_OUTREQCNTfield (0xFF), till it is zero.
 *9. write 1 to LW_MC_CLIENT_HOTRESETN_0 bit 4 to disable clear of EPP requests inside MC.
 *10.write 1 to LW_MC_CLIENT_CTRL_0 bit 4 to unblock EPP request inside MC.
 *11.bring EPP out of reset by write 1 to bit 6 of LW_HOST1X_ASYNC_RSTREG_0 (offset 0x14). Do read
 *   modify write.
 *12.program LW_EPP_CTXSW_0 (0x0) with previously saved EPP_CTXSW_DATA.
 *13.program EPP registers
 *14.enable encoder
 *15.enable VI EPP output by programming bit 4:2 to the desired format of LW_VI_VI_CORE_CONTROL_0 (offset 0x3)
 *
 * latest comment about EPP output buffer by Ignatius in 4-25-07:
 * There is also some use-case information in  *hw/ar/doc/sc15/sc15.doc chapter 1.3.
 *Theoretically 2 output buffers should be sufficient if there is no EPP overrun. At one time we suspected
 * overrun and we change recommendation to 3 buffers in one of the bug (Rakesh filed the S/W equivalent) but
 * it turns out that the bug was caused by memory issue in 2MI part.  But this memory issue and it is supposedly
 * fixed in A02 version. I checked all SC15 ext mem controller bugs and can't see anything that hasn't been fixed
 * that could cause issue.  So, I suppose it's OK to use DRAM for EPP buffers.  The only constraint is bandwidth.
 * 2MI parts are pretty poor in bandwidth and high res sensor have high pixel rate so you must sustain both
 * writing to memory as well as JPEG encoder reading from memory.
 * AP15 flow control
 * For AP15, we will not use host1xRDMA engine. EPP will write into output buffer as is. SW will
 * use SYNCPT_OP_DONE as an indicator of one EPP output buffer is ready for read. After SW
 * consumes one buffer, SW will write to BUFFER_RELEASE register.
 * EPP has an internal counter. Every buffer filed will increament the buffer, every write from
 * SW to BUFFER_RELEASE register will decrease this counter. EPP will stall input bus if:
 * counter >= EPP_OUTPUT_BUFFER_COUNT - 1
 * For SW to use this flow control correctly, SW has to release all the buffers that locked by
 * EPP to maintain synchronizations of SW and EPP. For example, after flow control is enabled,
 * EPP output 4 buffers, our flowControlBufferCount = 4. SW only need 2 of them. SW should
 * write to BUFFER_RELEASE register 4 times before switch EPP for other stream capturing. There
 * is no reset of this counter or wrap around. This buffer will be zero after reset. EPP
 * RTL does provide a EPP_DEBUG_CONTROL_FLOW_COUNTER register, but it is for debug only.
 * added syncpt mail from Bruce: Sent: Sun 8/12/2007 9:19 PM
 * Subject: Clarification on syncpt related limitation for SB / EPP
 * We dislwssed this so many months ago that it should be mentioned again
 * (Kevin ran across this Friday with SB/EPP randoms).
 * When using SB with EPP, both must be programmed so that the unit of work
 * represented by OP_DONE is the same.  For SB, the unit of work is a single BLT.
 * For EPP it is processing of one buffer (you program a control register with
 * the number of lines in a buffer).  So if you process a whole frame with
 * one SB BLT and program EPP so that a buffer == frame, everything is OK.
 * The problem comes when you do something like program SB to process a
 * whole frame, but program EPP to store that as two buffers.
 * I not aware of any problems that this constraint might cause (I've dislwssed
 * this privately with MattL, Patrick, and David).
 * If there are any strong feelings in the SW group about this matter, then
 * a WNF bug should be opened to add to the to-do list for AP20.
 * EPP syncpt can be enabled through host interface or through syncpt trigger bus. For
 * AP15, syncpt trigger bus is the only recommended way to enable EPP syncpt. None of
 * the host interface trigger is fully verified and hense, the behavior is not guaranteed
 * to be as specified.
 *
 */

#define LWE2B5_INCR_SYNCPT_0                               (0x0)
// Condition mapped from raise/wait
#define LWE2B5_INCR_SYNCPT_0_COND                          15:8
#define LWE2B5_INCR_SYNCPT_0_COND_IMMEDIATE                (0)
#define LWE2B5_INCR_SYNCPT_0_COND_OP_DONE                  (1)
#define LWE2B5_INCR_SYNCPT_0_COND_RD_DONE                  (2)
#define LWE2B5_INCR_SYNCPT_0_COND_REG_WR_SAFE              (3)
#define LWE2B5_INCR_SYNCPT_0_COND_COND_4                   (4)
#define LWE2B5_INCR_SYNCPT_0_COND_COND_5                   (5)
#define LWE2B5_INCR_SYNCPT_0_COND_COND_6                   (6)
#define LWE2B5_INCR_SYNCPT_0_COND_COND_7                   (7)
#define LWE2B5_INCR_SYNCPT_0_COND_COND_8                   (8)
#define LWE2B5_INCR_SYNCPT_0_COND_COND_9                   (9)
#define LWE2B5_INCR_SYNCPT_0_COND_COND_10                  (10)
#define LWE2B5_INCR_SYNCPT_0_COND_COND_11                  (11)
#define LWE2B5_INCR_SYNCPT_0_COND_COND_12                  (12)
#define LWE2B5_INCR_SYNCPT_0_COND_COND_13                  (13)
#define LWE2B5_INCR_SYNCPT_0_COND_COND_14                  (14)
#define LWE2B5_INCR_SYNCPT_0_COND_COND_15                  (15)

// syncpt index value
#define LWE2B5_INCR_SYNCPT_0_INDX                          7:0

#define LWE2B5_INCR_SYNCPT_CNTRL_0                         (0x1)
// If NO_STALL is 1, then when fifos are full,
// INCR_SYNCPT methods will be dropped and the
// INCR_SYNCPT_ERROR[COND] bit will be set.
// If NO_STALL is 0, then when fifos are full,
// the client host interface will be stalled.
#define LWE2B5_INCR_SYNCPT_CNTRL_0_INCR_SYNCPT_NO_STALL    8:8

// If SOFT_RESET is set, then all internal state
// of the client syncpt block will be reset.
// To do soft reset, first set SOFT_RESET of
// all host1x clients affected, then clear all
// SOFT_RESETs.
#define LWE2B5_INCR_SYNCPT_CNTRL_0_INCR_SYNCPT_SOFT_RESET  0:0


#define LWE2B5_INCR_SYNCPT_ERROR_0                         (0x2)
// COND_STATUS[COND] is set if the fifo for COND overflows.
// This bit is sticky and will remain set until cleared.
// Cleared by writing 1.
#define LWE2B5_INCR_SYNCPT_ERROR_0_COND_STATUS             31:0

// just in case names were redefined using macros
#define LW_EPP_INCR_SYNCPT_NB_CONDS                     4
//COND_FIFO_DEPTH (ignored for COND 0)
//COND_TRIG_MODE applies to COND 1..x (ignored for 0,1)

#define LWE2B5_EPP_SYNCPT_DEST_0                           (0x8)
#define LWE2B5_EPP_SYNCPT_DEST_0_HOST                       0:0

#define LWE2B5_EPP_SYNCPT_DEST_0_MPE                       1:1

// Context switch reg is defined in this include file which takes one 32-bit register space
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

#define LWE2B5_CTXSW_0                                 (0x9)
// Current working class
#define LWE2B5_CTXSW_0_LWRR_CLASS                      9:0

// Automatically acknowledge any incoming context switch requests
#define LWE2B5_CTXSW_0_AUTO_ACK                        11:11
#define LWE2B5_CTXSW_0_AUTO_ACK_MANUAL                 (0)
#define LWE2B5_CTXSW_0_AUTO_ACK_AUTOACK                (1)

// Current working channel, reset to 'invalid'
#define LWE2B5_CTXSW_0_LWRR_CHANNEL                    15:12

// Next requested class
#define LWE2B5_CTXSW_0_NEXT_CLASS                      25:16

// Next requested channel
#define LWE2B5_CTXSW_0_NEXT_CHANNEL                    31:28


// This reflects status of all pending
//  interrupts which is valid as long as
//  the interrupt is not cleared even if the
//  interrupt is masked. A pending interrupt
//  can be cleared by writing a '1' to
//  the corresponding interrupt status bit
//  in this register.
#define LWE2B5_INTSTATUS_0                             (0xa)
// Context Switch Interrupt Status (this is
//  cleared on write).
#define LWE2B5_INTSTATUS_0_CTXSW_INT                   0:0
#define LWE2B5_INTSTATUS_0_CTXSW_INT_NOTPENDING        (0)    // //  interrupt not pending
#define LWE2B5_INTSTATUS_0_CTXSW_INT_PENDING           (1)    // //  interrupt pending


// Frame End Interrupt Status.
// If enabled, interrupt is generated everytime
//  FRAME_HEIGHT count expires after all frame
//  data is written to memory.
#define LWE2B5_INTSTATUS_0_FRAME_END_INT               1:1
#define LWE2B5_INTSTATUS_0_FRAME_END_INT_NOTPENDING    (0)    // //  interrupt not pending
#define LWE2B5_INTSTATUS_0_FRAME_END_INT_PENDING       (1)    // //  interrupt pending


// Output Buffer End Interrupt Status.
// If enabled, interrupt is generated everytime
//  end of buffer is reached. Typically this is
//  determined when OB0_V_SIZE count expires
//  and after all buffer data is written to
//  memory. Note that buffer may not be
//  completely filled for first and last buffer
//  of each frame. This interrupt should be
//  used together with LAST_BUFFER_INDEX status.
#define LWE2B5_INTSTATUS_0_BUFFER_END_INT                   2:2
#define LWE2B5_INTSTATUS_0_BUFFER_END_INT_NOTPENDING       (0)    // //  interrupt not pending
#define LWE2B5_INTSTATUS_0_BUFFER_END_INT_PENDING          (1)    // //  interrupt pending


// Short Frame Interrupt Status.
// If enabled, interrupt is generated when
//  shorter than expected frame is detected.
//  Input stream end-of-frame flag and
//  OUTPUT_FRAME_SIZE is used to determine
//  this condition.
#define LWE2B5_INTSTATUS_0_SHORT_FRAME_INT                 8:8
#define LWE2B5_INTSTATUS_0_SHORT_FRAME_INT_NOTPENDING      (0)    // //  interrupt not pending
#define LWE2B5_INTSTATUS_0_SHORT_FRAME_INT_PENDING         (1)    // //  interrupt pending

// Last buffer index status.
// This indicates the index of the previous
//  (last) buffer written to memory by EPP.
#define LWE2B5_INTSTATUS_0_LAST_BUFFER_INDEX                31:24

//  This specifies processing control for
#define LWE2B5_EPP_CONTROL_0                               (0xb)
// Input source.
#define LWE2B5_EPP_CONTROL_0_INPUT_SOURCE                   1:0
#define LWE2B5_EPP_CONTROL_0_INPUT_SOURCE_VI               (0)    // // Input source from VI.
#define LWE2B5_EPP_CONTROL_0_INPUT_SOURCE_SB               (1)    // // Input source from StretchBLT (2D).
#define LWE2B5_EPP_CONTROL_0_INPUT_SOURCE_DISPLAY          (2)    // // Input source from DISPLAY.
#define LWE2B5_EPP_CONTROL_0_INPUT_SOURCE_DISPLAYB         (3)    // // Input source from DISPLAY B


// Color Space Colwerter enable.
//  This enables RGB to YUV color space
//  colwerter and should be enabled only for
//  RGB input format if output format is YUV.
//  If enabled, the output of the color space
//  colwerter is YUV444 therefore colwersion
//  to YUV422 or YUV420 may be needed also.
#define LWE2B5_EPP_CONTROL_0_ENABLE_CC                     2:2
#define LWE2B5_EPP_CONTROL_0_ENABLE_CC_DISABLE             (0)    // // RGB input format is not colwerted to YUV.
#define LWE2B5_EPP_CONTROL_0_ENABLE_CC_ENABLE              (1)    // // RGB input format is colwerted to YUV.


// YUV444 to YUV422 enable.
//  If set to ENABLE, chroma data is
//  horizontally reduced by half.
//  YUV444 to YUV422 colwersion must be
//  enabled when OUTPUT_FORMAT is YUV422 or
//  YUV420.
//  YUV444 to YUV422 colwersion must be
//  disabled when OUTPUT_FORMAT is YUV422R.
//  The horizontal chroma reduction method is
//  controlled by CHROMA_FILTER_422 field.
#define LWE2B5_EPP_CONTROL_0_ENABLE_422                     3:3
#define LWE2B5_EPP_CONTROL_0_ENABLE_422_DISABLE                    (0)    // // YUV444 to YUV422 colwersion is disabled.
#define LWE2B5_EPP_CONTROL_0_ENABLE_422_ENABLE                     (1)    // // YUV444 to YUV422 colwersion is enabled.


// YUV422 to YUV420 enable.
//  If set to ENABLE, chroma data is
//  vertically reduced by half.
//  YUV422 to YUV420 colwersion must be
//  enabled when OUTPUT_FORMAT is YUV420 and
//  input format is not MIPI CSI YUV420.
//  YUV422 to YUV420 colwersion must also be
//  enabled when OUTPUT_FORMAT is YUV422R.
//  The vertical chroma reduction method is
//  controlled by CHROMA_FILTER_420 field.
#define LWE2B5_EPP_CONTROL_0_ENABLE_420                     4:4
#define LWE2B5_EPP_CONTROL_0_ENABLE_420_DISABLE                    (0)    // // YUV422 to YUV420 colwersion is disabled.
#define LWE2B5_EPP_CONTROL_0_ENABLE_420_ENABLE                     (1)    // // YUV422 to YUV420 colwersion is enabled.


// Luma PreProcess filter enable.
//  This controls luma pre-encoding filter for
//  YUV output formats including YUV420 when
//  input format is YUV420 CSI format.
//  Enabling the luma pre-encoding filter will
//  reduce noise in the input stream and may
//  result in more compact encoding.
#define LWE2B5_EPP_CONTROL_0_ENABLE_PP                      5:5
#define LWE2B5_EPP_CONTROL_0_ENABLE_PP_DISABLE                     (0)    // // Luma pre-encoding filter is disabled.
#define LWE2B5_EPP_CONTROL_0_ENABLE_PP_ENABLE                      (1)    // // Luma pre-encoding filter is enabled.


// Chroma YUV422 to YUV420 colwersion.
//  This controls chroma reduction when YUV422
//  to YUV420 colwersion is enabled.
//  This is effective only when ENABLE_420 is
//  set to ENABLE.
#define LWE2B5_EPP_CONTROL_0_CHROMA_FILTER_420                      7:6
#define LWE2B5_EPP_CONTROL_0_CHROMA_FILTER_420_DROP                (0)    // // Even chroma lines are dropped.
#define LWE2B5_EPP_CONTROL_0_CHROMA_FILTER_420_AVERAGE             (1)    // // Averaging is done for each pair (odd and
//  even) of chroma lines.


// Non-planar YUV422 output format.
//  This selects output format when output
//  format is set to YUV422 non-planar.
#define LWE2B5_EPP_CONTROL_0_OUTPUT_NP_FORMAT                       9:8
#define LWE2B5_EPP_CONTROL_0_OUTPUT_NP_FORMAT_UYVY                 (0)    // // UY0VY1, where U is LSB and Y1 is MSB
#define LWE2B5_EPP_CONTROL_0_OUTPUT_NP_FORMAT_VYUY                 (1)    // // VY0UY1, where V is LSB and Y1 is MSB
#define LWE2B5_EPP_CONTROL_0_OUTPUT_NP_FORMAT_YUYV                 (2)    // // Y0UY1V, where Y0 is LSB and V is MSB
#define LWE2B5_EPP_CONTROL_0_OUTPUT_NP_FORMAT_YVYU                 (3)    // // Y0VY1U, where Y0 is LSB and U is MSB


// Output planar format.
//  This should be set to ENABLE if output
//  format is YUV planar or semi-planar
//  and should be DISABLE if output format
//  is YUV non-planar.
//  For RGB output data formats, this bit has
//  alternate function of swapping R and B.
#define LWE2B5_EPP_CONTROL_0_OUTPUT_PLANAR                         10:10
#define LWE2B5_EPP_CONTROL_0_OUTPUT_PLANAR_DISABLE                 (0)    // // For YUV output formats, output is YUV
//  non-planar format.
// For RGB output formats, R is in upper bits
//  and B is in lower bits.
#define LWE2B5_EPP_CONTROL_0_OUTPUT_PLANAR_ENABLE                  (1)    // // Output is YUV planar/semi-planar format.
// For RGB output formats, B is in upper bits
//  and R is in lower bits.


// Chroma YUV444 to YUV422 colwersion.
//  This controls chroma reduction when YUV444
//  to YUV422 colwersion is enabled.
//  This is effective only when ENABLE_422 is
//  set to ENABLE.
#define LWE2B5_EPP_CONTROL_0_CHROMA_FILTER_422                      11:11
#define LWE2B5_EPP_CONTROL_0_CHROMA_FILTER_422_DROP                 (0)    // // Even chroma pixels are dropped. This may be
//  used if the incoming YUV444 stream has
//  chroma pixel duplication of odd and even
//  pixels.

#define LWE2B5_EPP_CONTROL_0_CHROMA_FILTER_422_FILTER              (1)    // // Horizontal chroma filtering is applied for
//  YUV444 to YUV422 colwersion.


// Output data format.
//  This specifies output data format together
//  with OUTPUT_FORMAT_EXT.
#define LWE2B5_EPP_CONTROL_0_OUTPUT_FORMAT                         14:12
#define LWE2B5_EPP_CONTROL_0_OUTPUT_FORMAT_YUV420                  (0)    // // YUV420 planar/semi-planar.
//  ENABLE_422 must be set to ENABLE for this
//  output format.
//  ENABLE_420 must be set to ENABLE if the
//  input format is not MIPI CSI YUV420 formats
//  and must be set to DISABLE if the input
//  format is MIPI CSI YUV420 formats.

#define LWE2B5_EPP_CONTROL_0_OUTPUT_FORMAT_YUV422                  (1)    // // YUV422 non-planar/planar/semi-planar.
//  If YUV422 planar/semi-planar format is
//  selected and XY swap is enabled, then
//  output will be written to memory in YUV422R
//  planar/semi-planar format correspondingly.
//  YUV422 non-planar format cannot be used with
//  XY swap enabled.
//  ENABLE_422 must be set to ENABLE and
//  ENABLE_420 must be set to DISABLE for this
//  output format.

#define LWE2B5_EPP_CONTROL_0_OUTPUT_FORMAT_YUV422R                 (2)    // // YUV422R planar/semi-planar.
//  If YUV422R planar/semi-planar format is
//  selected and XY swap is enabled, then
//  output will be written to memory in YUV422
//  planar/semi-planar format correspondingly.
//  ENABLE_422 must be set to DISABLE and
//  ENABLE_420 must be set to ENABLE for this
//  output format.

#define LWE2B5_EPP_CONTROL_0_OUTPUT_FORMAT_YUV444                  (3)    // // AYUV444 non-planar if OUTPUT_PLANAR=DISABLE.
#define LWE2B5_EPP_CONTROL_0_OUTPUT_FORMAT_RGB888                  (4)    // // ARGB888
#define LWE2B5_EPP_CONTROL_0_OUTPUT_FORMAT_RGBRAW                  (5)    // // RGBRAW
#define LWE2B5_EPP_CONTROL_0_OUTPUT_FORMAT_BAYERRAW                (6)    // // BAYERRAW


// Chroma data sign.
//  This indicates chroma data format.
#define LWE2B5_EPP_CONTROL_0_CHROMA_SIGN                           15:15
#define LWE2B5_EPP_CONTROL_0_CHROMA_SIGN_UNSIGNED                  (0)    // // unsigned chroma data, Cb/Cr
#define LWE2B5_EPP_CONTROL_0_CHROMA_SIGN_SIGNED                    (1)    // // signed chroma data, U/V, in 2's complement


// DMA enable.
//  S/W should program this bit to a 1 every time
//  DMA is enabled and buffer configuration changes
//  This bit is now sent out as init for all the host
//  counters and logic.
//  This enables EPP trigger at end of each
//  buffer to be sent to Read DMA. The Read DMA
//  must be properly programmed to fetch EPP
//  buffers in memory upon receiving these
//  triggers.
#define LWE2B5_EPP_CONTROL_0_DMA_ENABLE                            16:16
#define LWE2B5_EPP_CONTROL_0_DMA_ENABLE_DISABLE                    (0)    // // EPP trigger to Read DMA is disabled
#define LWE2B5_EPP_CONTROL_0_DMA_ENABLE_ENABLE                     (1)    // // EPP trigger to Read DMA is enabled


// Pixel duplication at start and end of lines.
//  This enables duplication of first pixel and
//  last pixel of each line to 128-bit boundary
//  if there is unused space in the 128-bit
//  memory word that contains the first and
//  last pixel. This may be enabled if output
//  of EPP is used as input of JPEG encoder.
//  JPEGE encoder can duplicate lines to ensure
//  JPEGE has a full MLW to encode.
//  If EPP Is used as input to MPEG encoder,
//  output height of EPP has to be set to multiples
//  of 16 for MPEG encoder does not duplicate lines.
//  This is only applicable for planar YUV
//  output formats (OUTPUT_PLANAR=ENABLE and
//  outputFormat!=RAWRGB and outputFormat!=RGB888).
//  Pixel duplication oclwrs prior to XY swap
//  process therefore if XY_SWAP is set to
//  ENABLE, the duplicated pixels will be
//  placed on top and bottom of the image
//  instead of left and right of the image and
//  therefore may not help in JPEG encoding
//  process. If image is to be rotated prior to
//  JPEG encoding, it is preferred to do the
//  rotation in the JPEG encoder rather than
//  in EPP.
#define LWE2B5_EPP_CONTROL_0_ENABLE_DUP                            17:17
#define LWE2B5_EPP_CONTROL_0_ENABLE_DUP_DISABLE                    (0)    // // First and last pixel duplication is disabled.
#define LWE2B5_EPP_CONTROL_0_ENABLE_DUP_ENABLE                     (1)    // // First and last pixel duplication is enabled
//  for each line.


// Output data format extension.
//  This specifies output data format together
//  with OUTPUT_FORMAT.
//  RGB formats are specified from lsb to msb.
#define LWE2B5_EPP_CONTROL_0_OUTPUT_FORMAT_EXT                      21:18
#define LWE2B5_EPP_CONTROL_0_OUTPUT_FORMAT_EXT_EXT0                        (0)    // // YUV420:  planar
// YUV422:  non-planar if OUTPUT_PLANAR=DISABLE,
//          planar if OUTPUT_PLANAR=ENABLE.
// YUV422R: planar
// YUV444:  AYUV444 non-planar if
//          OUTPUT_PLANAR=DISABLE
// RGB888:  B8G8R8A8 if OUTPUT_PLANAR=DISABLE,
//          R8G8B8A8 if OUTPUT_PLANAR=ENABLE.
// RGBRAW:  B5G6R5 if OUTPUT_PLANAR=DISABLE,
//          R5G6B5 if OUTPUT_PLANAR=ENABLE.
// BAYERRAW: 2-byte bayer/raw

#define LWE2B5_EPP_CONTROL_0_OUTPUT_FORMAT_EXT_EXT1                        (1)    // // YUV420:  semi-planar
// YUV422:  semi-planar if OUTPUT_PLANAR=ENABLE.
// YUV422R: semi-planar
// RGB888:  A8B8G8R8 if OUTPUT_PLANAR=DISABLE,
//          A8R8G8B8 if OUTPUT_PLANAR=ENABLE.
// RGBRAW:  A1B5G5R5 if OUTPUT_PLANAR=DISABLE,
//          A1R5G5B5 if OUTPUT_PLANAR=ENABLE.
// BAYERRAW: 1-byte bayer/raw

#define LWE2B5_EPP_CONTROL_0_OUTPUT_FORMAT_EXT_EXT2                        (2)    // // YUV420:  planar, CSI YUV420 legacy input
// RGBRAW:  B5G5R5A1 if OUTPUT_PLANAR=DISABLE,
//          R5G5B5A1 if OUTPUT_PLANAR=ENABLE.

#define LWE2B5_EPP_CONTROL_0_OUTPUT_FORMAT_EXT_EXT3                        (3)    // // YUV420:  semi-planar, CSI YUV420 legacy input
// RGBRAW:  A4B4G4R4 if OUTPUT_PLANAR=DISABLE,
//          A4R4G4B4 if OUTPUT_PLANAR=ENABLE.

#define LWE2B5_EPP_CONTROL_0_OUTPUT_FORMAT_EXT_EXT4                        (4)    // // YUV420:  planar, CSI YUV420 input.
// RGBRAW:  B4G4R4A4 if OUTPUT_PLANAR=DISABLE,
//          R4G4B4A4 if OUTPUT_PLANAR=ENABLE.

#define LWE2B5_EPP_CONTROL_0_OUTPUT_FORMAT_EXT_EXT5                        (5)    // // YUV420:  semi-planar, CSI YUV420 input.


// enable SW flow control default is disabled.
#define LWE2B5_EPP_CONTROL_0_SW_FLOW_CONTROL                24:24
#define LWE2B5_EPP_CONTROL_0_SW_FLOW_CONTROL_DISABLE                       (0)
#define LWE2B5_EPP_CONTROL_0_SW_FLOW_CONTROL_ENABLE                        (1)


//  This specifies width and height of input
//  frame with respect to size of active area
//  to be processed by EPP.
//  For some planar and semi-planar YUV
//  output formats, there are 2:1 ratio of luma
//  vs chroma pixels horizontally and/or
//  vertically. In this case, output frame
//  size specifies luma plane size and should
//  be programmed to even values if the
//  corresponding chroma plane dimension is
//  half of the luma plane dimension.
#define LWE2B5_OUTPUT_FRAME_SIZE_0                 (0xc)
// Frame width in pixels (min 16 pixels).
#define LWE2B5_OUTPUT_FRAME_SIZE_0_FRAME_WIDTH                      15:0

// Frame height in lines (min 1 line).
#define LWE2B5_OUTPUT_FRAME_SIZE_0_FRAME_HEIGHT                     31:16


//  This specifies position of the first pixel
//  in the input frame to be processed by EPP.
//  Together with output frame size, this
//  parameter defines the active size of the
//  input surface.
#define LWE2B5_INPUT_FRAME_AOI_0                   (0xd)
// Horizontal offset in term of pixel position
//  (min 0).
#define LWE2B5_INPUT_FRAME_AOI_0_FRAME_HORI_OFFSET                  15:0

// Vertical offset in term of line position
//  (min 0).
#define LWE2B5_INPUT_FRAME_AOI_0_FRAME_VERT_OFFSET                  31:16

// This register can be used to program the output frame orientation as follows:
//   XY_SWAP | VERT_DIR | HORI_DIR | Output frame orientation
// ----------|----------|----------|------------------------------------------------------
//      0    |    0     |    0     | Normal (same as input frame orientation)
//      0    |    0     |    1     | H flip (mirror on vertical axis)
//      0    |    1     |    0     | V flip (mirror on horizontal axis)
//      0    |    1     |    1     | 180-degree rotation
//      1    |    0     |    0     | XY swap (mirror on 315-degree diagonal)
//      1    |    0     |    1     | 270-degree rotation
//      1    |    1     |    0     | 90-degree rotation
//      1    |    1     |    1     | XY swap and H,V flips (mirror on 45-degree diagonal)
// ---------------------------------------------------------------------------------------
// Notes:
// a. XY swap, if enabled, is performed first prior to H-flip and V-flip.
// b. Output buffer start address should be programmed in consideration of the scan directions.
//    If xy-swap=0 (XY swap disabled):
//       OB0_Start_Address = OBSA + hsd * (width * bpp - 1) + vsd * (height - 1) * line_stride,
//       where line_stride >= ((width  * bpp) + 15) & 0xFFFFFFF0).
//    If xy-swap=1 (XY swap enabled):
//       OB0_Start_Address = OBSA + hsd * (height * bpp - 1) + vsd * (width - 1) * line_stride,
//       where line_stride >= ((height * bpp) + 15) & 0xFFFFFFF0).
//    In the above formulae, OBSA is the start address of output buffer 0, hsd and vsd
//    are H/V scan direction, bpp is byte per pixel, width and height are the sizes of input
//    image in pixels, and line_stride is the line stride of output buffer in bytes.
// c. For YUV422 non-planar, since chroma data is shared by multiple pixels, XY swap can NOT
//    be performed.

#define LWE2B5_OUTPUT_SCAN_DIR_0                   (0xe)
// Horizontal scan direction.
#define LWE2B5_OUTPUT_SCAN_DIR_0_HORI_DIR                   0:0
#define LWE2B5_OUTPUT_SCAN_DIR_0_HORI_DIR_INCREASE                 (0)    // // Increasing address.

#define LWE2B5_OUTPUT_SCAN_DIR_0_HORI_DIR_DECREASE                 (1)    // // Decreasing address.


// Vertical scan direction.
#define LWE2B5_OUTPUT_SCAN_DIR_0_VERT_DIR                   1:1
#define LWE2B5_OUTPUT_SCAN_DIR_0_VERT_DIR_INCREASE                 (0)    // // Increasing address.

#define LWE2B5_OUTPUT_SCAN_DIR_0_VERT_DIR_DECREASE                 (1)    // // Decreasing address.


// XY_SWAP IS NO LONGER SUPPORTED
#define LWE2B5_OUTPUT_SCAN_DIR_0_XY_SWAP                    2:2
#define LWE2B5_OUTPUT_SCAN_DIR_0_XY_SWAP_DISABLE                   (0)    // // XY swap disabled.

#define LWE2B5_OUTPUT_SCAN_DIR_0_XY_SWAP_ENABLE                    (1)    // // XY swap enabled.


// If output of EPP is used for input for JPEGE (JPEG encoder), it is preferred to have the
// same buffer accessing sequence for the two modules. Otherwise, it is hard to control the
// buffer between the two modules.
// Output buffer start addresses define the start address of the first buffer (buffer index 0)
// and taking into account the output orientation. Note that since output buffer start
// addresses depend on output orientation, therefore it must be reprogrammed when output
// orientation is changed.
// All addresses must be on pixel boundary.
// Output format is either Planar or Non-planar format. For non-planar output format, only
// luma buffer is used.
// For semi-planar output formats, only 2 planes are used. Chroma data is interleaved in a
// single plane. So for semi-planar outputs, only OB0_START_ADDRESS_U is used to specify
// chroma buffer start address. Enough memory should be reserved to store U and V data for
// the chroma buffer.

//  start address for non-planar output format.
#define LWE2B5_OB0_START_ADDRESS_Y_0                       (0xf)
// OB0 Y start address.
#define LWE2B5_OB0_START_ADDRESS_Y_0_OB0_START_ADDRESS_Y                    31:0


#define LWE2B5_OB0_BASE_ADDRESS_Y_0                        (0x10)
// OB0 Y BASE address.
#define LWE2B5_OB0_BASE_ADDRESS_Y_0_OB0_BASE_ADDRESS_Y                      31:0


#define LWE2B5_OB0_START_ADDRESS_U_0                       (0x11)
// OB0 U start address.
#define LWE2B5_OB0_START_ADDRESS_U_0_OB0_START_ADDRESS_U                    31:0


//  This is used to specify U buffer base
//  address for planar output format.
//  This is also used for semi-planar chroma
//  buffer base address.
//  This is not used for non-planar output
//  format.
#define LWE2B5_OB0_BASE_ADDRESS_U_0                        (0x12)
// OB0 U start address.
#define LWE2B5_OB0_BASE_ADDRESS_U_0_OB0_BASE_ADDRESS_U                      31:0


//  This is used to specify V buffer start
//  start address for planar output format.
//  This is not used for semi-planar and for
//  non-planar output format.
#define LWE2B5_OB0_START_ADDRESS_V_0                       (0x13)
// OB0 V start address.
#define LWE2B5_OB0_START_ADDRESS_V_0_OB0_START_ADDRESS_V                    31:0


//  This is used to specify V buffer base
//  address for planar output format.
//  This is not used for semi-planar and for
//  non-planar output format.
#define LWE2B5_OB0_BASE_ADDRESS_V_0                        (0x14)
// OB0 V start address.
#define LWE2B5_OB0_BASE_ADDRESS_V_0_OB0_BASE_ADDRESS_V                      31:0


// The value is epxressed in PIXELS
// For non-rotated surfaces, this points
// to the TOP-LEFT of the image.
#define LWE2B5_OB0_XY_OFFSET_LUMA_0                        (0x15)
// X offset in pixels for the first pixel
// written in the image in the Y buffer
#define LWE2B5_OB0_XY_OFFSET_LUMA_0_START_X_LUMA                    15:0

// Y offset in pixels for the first pixel
// in the image in the Y buffer.
#define LWE2B5_OB0_XY_OFFSET_LUMA_0_START_Y_LUMA                    31:16


// The value is epxressed in PIXELS
// For non-rotated surfaces, this points
// to the TOP-LEFT of the image.
#define LWE2B5_OB0_XY_OFFSET_CHROMA_0                      (0x16)
// X offset in pixels for the first pixel
// written in the image in the U and V buffer
// For YUV422R or YUV420P with Chroma Averaging on
// START_X_CHROMA must be aligned to a 16 byte boundary.
#define LWE2B5_OB0_XY_OFFSET_CHROMA_0_START_X_CHROMA                15:0

// Y offset in pixels for the first pixel
// in the image in the U and V buffer.
#define LWE2B5_OB0_XY_OFFSET_CHROMA_0_START_Y_CHROMA                31:16


//  This specifies the number of output buffers
//  and the vertical size of each buffer.
//  Note that horizontal size of each buffer
//  is implied from OB_LINE_STRIDE.
#define LWE2B5_OB0_SIZE_0                  (0x17)
// Output buffer count.
//  This specifies the number of buffers in
//  output buffer set 0.
#define LWE2B5_OB0_SIZE_0_OB0_COUNT                 7:0

// Output buffer vertical size.
//  This specifies the number of lines.
//  In the case of xyswap, vertical size should
//  be programmed as vertical size before xyswap.
#define LWE2B5_OB0_SIZE_0_OB0_V_SIZE                28:16


// Line stride should be programmed taking into
// account XY swap. If XY swap is enabled,
// line stride should be programmed as after xyswap.
#define LWE2B5_OB0_LINE_STRIDE_L_0                 (0x18)
// Output buffer luma line stride.
// This parameter must be programmed as 16-byte
//  multiple so 4 lsbs must be set to zeros.
#define LWE2B5_OB0_LINE_STRIDE_L_0_OB0_LINE_STRIDE_L                15:0

// Output buffer chroma line stride.
// This parameter must be programmed as 16-byte
//  multiple so 4 lsbs must be set to zeros.
#define LWE2B5_OB0_LINE_STRIDE_L_0_OB0_LINE_STRIDE_C                31:16

// Following two registers, OB0_BUFFER_STRIDE_LUMA and OB0_BUFFER_STRIDE_CHROMA are absolete for AP15
// due to changing memory addressing from linear to xy.
// Start today, 5-25-07, EPP rtl will not use the value programmed into these registers.
// As of today, 5-25-07, EPP cmod will still use these two registers.
// If SW driver wants to use cmod, you will have to program the registers as:
// OB0_BUFFER_STRIDE_LUMA = OB0_LINE_STRIDE_L * OB0_V_SIZE;
// OB0_BUFFER_STRIDE_CHROMA = OB0_LINE_STRIDE_C * OB0_V_SIZE/factor;
//                            factor = 1 for YUV422R and 2 for YUV420 and YUV422
// Yuanyuan will update epp cmodel as soon as VI and EPP verification is done, around Aug.

#define LWE2B5_OB0_BUFFER_STRIDE_LUMA_0                    (0x19)
// Output buffer luma buffer stride.
// This parameter must be programmed as 16-byte
//  multiple so 4 lsbs must be set to zeros.
#define LWE2B5_OB0_BUFFER_STRIDE_LUMA_0_OB0_BUFFER_STRIDE_L                 31:0


#define LWE2B5_OB0_BUFFER_STRIDE_CHROMA_0                  (0x1a)
// Output buffer chroma buffer stride.
// This parameter must be programmed as 16-byte
//  multiple so 4 lsbs must be set to zeros.
#define LWE2B5_OB0_BUFFER_STRIDE_CHROMA_0_OB0_BUFFER_STRIDE_C                       31:0

// Color Space Colwerter related registers are CSC_RGB2Y_COEFF, CSC_RGB2U_COEFF,
// CSC_RGB2V_COEFF and CSC_YOFFSET_COEFF.
// The recommended coefficient values differ based on the type of colwersion either RGB to
// ITU-R BT.601 Y/Cb/Cr or RGB to ITU-R BT.709 Y/Cb/Cr is used. For both the type of
// colwersions, R,G,B values are expected to be in the range of [0,255] and after Color Space
// Colwersion operation, Y value is in the range of [16,235] with Y-Offset of 16, Cb and Cr are
// in the range of [16,240] with offset of 128. The Offset of 128 for Cb and Cr are hard-coded
// in the design.
// The coefficient for R component CSC_R2Y_COEFF is represented as Unsigned U0.8 (8-bit),
// the coefficient for G component CSC_G2Y_COEFF is represented as Unsigned U1.8 (9-bits),
// the coefficient for B component CSC_B2Y_COEFF is represented as Unsigned U0.8 (8-bits).
//
// For RGB to ITU-R BT.601 Y/Cb/Cr colwersion, the recommended coefficient values are:
// [CSC_R2Y_COEFF CSC_G2Y_COEFF CSC_B2Y_COEFF] = [0.256788 0.504129 0.097906]
// The coefficients are represented with 8-bit fraction (U1.8 or U0.8) therefore for a dynamic
// range of 256 (8-bits), the programmed coefficient values are [66 129 25] = [42h 81h 19h]
//
// For RGB to ITU-R BT.709 Y/Cb/Cr colwersion, the recommended coefficient values are:
// [CSC_R2Y_COEFF CSC_G2Y_COEFF CSC_B2Y_COEFF] = [0.182586 0.614231 0.062007]
// The coefficients are represented with 8-bit fraction (U1.8 or U0.8) therefore for a dynamic
// range of 256 (8-bits), the programmed coefficient values are [47 157 16] = [2fh 9dh 10h]
//

#define LWE2B5_CSC_RGB2Y_COEFF_0                   (0x1b)
// Coefficient for R to Y component.
#define LWE2B5_CSC_RGB2Y_COEFF_0_CSC_R2Y_COEFF                      7:0

// Coefficient for G to Y component.
#define LWE2B5_CSC_RGB2Y_COEFF_0_CSC_G2Y_COEFF                      16:8

// Coefficient for B to Y component.
#define LWE2B5_CSC_RGB2Y_COEFF_0_CSC_B2Y_COEFF                      24:17

// The coefficient for R component CSC_R2U_COEFF is represented as Sign Magnitude S0.8 (9-bits),
// the coefficient for G component CSC_G2U_COEFF is represented as Sign Magnitude S0.8 (9-bits),
// the coefficient for B component CSC_B2U_COEFF is represented as Sign Magnitude S0.8 (9-bits).
// The most significant bit (9th bit)  is either set to 0 or 1 based on the coefficient value
// being positive or negative respectively.
//
// For RGB to ITU-R BT.601 Y/Cb/Cr colwersion, the recommended coefficient values are:
// [CSC_R2U_COEFF CSC_G2U_COEFF CSC_B2U_COEFF] = [-0.148223 -0.290993 0.439216]
// The coefficients are represented with 8-bit fraction (S0.8) therefore for a dynamic range of
// 256 (8-bits), the programmed coefficient values are [-38 -74 112] = [-26h -4ah 70h] =
// [126h 14ah 70h]
//
// For RGB to ITU-R BT.709 Y/Cb/Cr colwersion, the recommended coefficient values are:
// [CSC_R2U_COEFF CSC_G2U_COEFF CSC_B2U_COEFF] = [-0.100644 -0.338572 0.439216]
// The coefficients are represented with 8-bit fraction (S0.8) therefore for a dynamic range of
// 256 (8-bits), the programmed coefficient values are [-26 -87 112] = [-1ah -57h 70h] =
// [11ah 157h 70h]
//

#define LWE2B5_CSC_RGB2U_COEFF_0                   (0x1c)
// Coefficient for R to U component.
#define LWE2B5_CSC_RGB2U_COEFF_0_CSC_R2U_COEFF                      8:0

// Coefficient for G to U component.
#define LWE2B5_CSC_RGB2U_COEFF_0_CSC_G2U_COEFF                      17:9

// Coefficient for B to U component.
#define LWE2B5_CSC_RGB2U_COEFF_0_CSC_B2U_COEFF                      26:18

// The coefficient for R component CSC_R2V_COEFF is represented as Sign Magnitude S0.8 (9-bits),
// the coefficient for G component CSC_G2V_COEFF is represented as Sign Magnitude S0.8 (9-bits),
// the coefficient for B component CSC_B2V_COEFF is represented as Sign Magnitude S0.8 (9-bits).
// The most significant bit (9th bit)  is either set to 0 or 1 based on the coefficient value
// being positive or negative respectively.
//
// For RGB to ITU-R BT.601 Y/Cb/Cr colwersion, the coefficient values recommended are:
// [CSC_R2V_COEFF CSC_G2V_COEFF CSC_B2V_COEFF] = [0.439216 -0.367788 -0.040274]
// The coefficients are represented with 8-bit fraction (S0.8) therefore for a dynamic range of
// 256 (8-bits), the programmed coefficient values are [112 -94 -18] = [70h -5eh -12h] =
// [70h 15eh 112h]
// For RGB to ITU-R BT.709 Y/Cb/Cr colwersion, the coefficient values recommended are:
// [CSC_R2V_COEFF CSC_G2V_COEFF CSC_B2V_COEFF] = [0.439216 -0.398942 -0.071427]
// The coefficients are represented with 8-bit fraction (S0.8) therefore for a dynamic range of
// 256 (8-bits), the programmed coefficient values are [112 -102 -10] = [70h -66h -0ah] =
// [70h 166h 10ah]

#define LWE2B5_CSC_RGB2V_COEFF_0                   (0x1d)
// Coefficient for R to V component.
#define LWE2B5_CSC_RGB2V_COEFF_0_CSC_R2V_COEFF                      8:0

// Coefficient for G to V component.
#define LWE2B5_CSC_RGB2V_COEFF_0_CSC_G2V_COEFF                      17:9

// Coefficient for B to V component.
#define LWE2B5_CSC_RGB2V_COEFF_0_CSC_B2V_COEFF                      26:18


#define LWE2B5_CSC_YOFFSET_COEFF_0                 (0x1e)
// Coefficient for Y Offset.
#define LWE2B5_CSC_YOFFSET_COEFF_0_CSC_YOFF_COEFF                   7:0

//reg YDATA_NOISE_CORING_THRESHOLD   incr1   // This threshold value is used in the Noise Filter
//    7:0  rw Y_NOISE_THRESH     i=0         // path of Y Data
//
//;
// There are two filters in EPP -- luma pre-processing filter and chroma 444 to 420 filter.
// Each filter is designed to be 7 tab, LPF.
//
// The recommended coefficients for luma pre-processing filter are: (0 1 4 6 4 1 0)/16.
// The filter coefficients are symmetric, therefore only four coefficients are programmed.
// The recommended coefficient base is 16 and programmed to 4, in terms of power of 2.
// The sum of all coefficients should not exceed the coefficient base value.
//
// The recommended coefficients for chroma 444-to-422 filter are: (-1 0 9 16 9 0 -1)/64.
// The filter coefficients are symmetric, therefore only four coefficients are programmed.
// The recommended coefficient base is 64 and programmed to 6, in terms of power of 2.
// The sum of all coefficients should not exceed the coefficient base value.
//
// The basic concept for this filter is:
//                      -------      low freq Data       ---
// inputData ---------->| LPF |------------------------>| + | ----> LPF filtered data
//               |      -------    |                     ---
//               |                 |                      ^
//               |                 |                      |
//               |                ---                  --------
//               --------------->| - |--------------->| coring |
//                                ---  high freq Data  --------
//

//  This specifies min/max values for filter
//  coring for both luma pre-processing filter
//  and chroma 444-to-422 filter.
#define LWE2B5_FILTER_BOUND_0                      (0x1f)
// Luma low bound
#define LWE2B5_FILTER_BOUND_0_LUMA_THRESHOLD_LOW                    7:0

// Luma high bound
#define LWE2B5_FILTER_BOUND_0_LUMA_THRESHOLD_HIGH                   15:8

// Chroma low bound
#define LWE2B5_FILTER_BOUND_0_CHROMA_THRESHOLD_LOW                  23:16

// Chroma high bound
#define LWE2B5_FILTER_BOUND_0_CHROMA_THRESHOLD_HIGH                 31:24


//
#define LWE2B5_FILTER_BASE_0                       (0x20)
// Luma pre-processing filter coefficient base.
//  This is a positive value programmed in terms
//  of power of 2.
#define LWE2B5_FILTER_BASE_0_PP_BASE                7:0

// Chroma 444-to-422 filter coefficient base.
//  This is a positive value programmed in terms
//  of power of 2.
#define LWE2B5_FILTER_BASE_0_CHROMA_BASE                    23:16

// Recommended luma pre-processing filter is a 5-tap filter (0,1,4,6,4,1,0)/16.
// An alternate recommendation is a 7-tap filter (1,6,15,20,15,6,1)/64.

//  This 7-tap filter is symmetric therefore
//  only four coefficients are programmed.
//  Sum of all 7 coefficients should not exceed
//  the filter base value.
#define LWE2B5_PP_FILTER_COEF_0                    (0x21)
// Luma pre-processing filter coefficients 1, 7.
//  This is a signed value -4 to +3.
#define LWE2B5_PP_FILTER_COEF_0_PP_COEF_0                   2:0

// Luma pre-processing filter coefficients 2, 6.
//  This is a signed value -16 to +15.
#define LWE2B5_PP_FILTER_COEF_0_PP_COEF_1                   7:3

// Luma pre-processing filter coefficients 3, 5.
//  This is a unsigned value 0 to 31.
#define LWE2B5_PP_FILTER_COEF_0_PP_COEF_2                   12:8

// Luma pre-processing filter coefficients 4.
//  This is an unsigned value 0 to 63
#define LWE2B5_PP_FILTER_COEF_0_PP_COEF_3                   18:13

// Recommended chroma 444-to-422 filter is a 5-tap filter (0,1,4,6,4,1,0)/16.
// An alternate recommendation is a 7-tap filter (-1,0,9,16,9,0,-1)/64.

//  This 7-tap filter is symmetric therefore
//  only four coefficients are programmed.
//  Sum of all 7 coefficients should not exceed
//  the filter base value.
#define LWE2B5_CHROMA_FILTER_COEF_0                        (0x22)
// Chroma 444-to-422 filter coefficients 1, 7.
//  This is a signed value -4 to +3.
#define LWE2B5_CHROMA_FILTER_COEF_0_CHROMA_COEF_0                   2:0

// Chroma 444-to-422 filter coefficients 2, 6.
//  This is a signed value -16 to +15.
#define LWE2B5_CHROMA_FILTER_COEF_0_CHROMA_COEF_1                   7:3

// Chroma 444-to-422 filter coefficients 3, 5.
//  This is a unsigned value 0 to 31.
#define LWE2B5_CHROMA_FILTER_COEF_0_CHROMA_COEF_2                   12:8

// Chroma 444-to-422 filter coefficients 4.
//  This is an unsigned value 0 to 63
#define LWE2B5_CHROMA_FILTER_COEF_0_CHROMA_COEF_3                   18:13


//  This is appended alpha valuve for RGB and
//  or YUV444 output data formats that requires
//  alpha. If the data format requires less than
//  8 bits of alpha then the necessary least
//  significant bits are used.
#define LWE2B5_ALFA_VALUE_0                        (0x23)
// Output alpha value.
#define LWE2B5_ALFA_VALUE_0_ALFA                    7:0

// Starting with SC17, U_LINE_BUFFER_ADDR will be use for chroma line buffer of both U and V
// data therefore V_LINE_BUFFER_ADDR is no longer used. The chroma buffer needs to sufficient
// for a whole chroma (U & V) line. U and V data will be byte-interleaved in this line buffer.

//  Chroma buffer needs to be allocated
//  whenever ENABLE_420 is set to ENABLE and
//  CHROMA_FILTER_420 is set to AVERAGE.
#define LWE2B5_U_LINE_BUFFER_ADDR_0                        (0x24)
// Chroma line buffer for U & V data.
//  This must be specified in 16-byte boundary
//  (the four lsbs must be set to zeros).
//  The size of this line buffer must be
//  sufficient to store one line of U & V.
#define LWE2B5_U_LINE_BUFFER_ADDR_0_U_LINE_BUFFER_ADDR                      31:0


#define LWE2B5_V_LINE_BUFFER_ADDR_0                        (0x25)
// Reserved.
#define LWE2B5_V_LINE_BUFFER_ADDR_0_V_LINE_BUFFER_ADDR                      31:0

// EPP may accept a raise from VI via the same input data bus from VI or two type of raises
//  from host.
// A raise from VI is returned when all preceding input data have been processed and written to
//  memory.
// Two type of raises maybe sent from host:
//  a. Raise Buffer
//  b. Raise Frame
// For raises from host, EPP tags last write for each buffer and each frame when sending it
//  to memory client and then it waits for the tag to be returned. Typically, raise buffer
//  will be acknowledged when the next end of buffer tag is returned by the memory write client.
//  Similary, typically, raise frame will be acknowledged when the next end of frame tag is
//  returned by the memory write client.
//  If raise buffer or raise frame is issued during vertical blank time (between end-of-frame
//  marker of previous frame and start-of-frame of next frame) then an option bit is added to
//  either acknowledge the raise immediately (SC15 compatible) or return the raise when the
//  next end of buffer or end of frame tag is returned by the memory write client
//  correspondingly.
//
// When one or more event that returns data to host are pending, priority will be assigned as
// follows:
//  (1) Context Switch Acknowledgement
//  (2) Raise from VI
//  (3) Raise Buffer
//  (4) Raise Frame
//  (5) Refcount or register read

//  The end of buffer event is generated when
//  the last data of the output buffer is
//  written to memory. This is independent of
//  output trigger so the end of buffer raise
//  acknowledge is returned even though there
//  maybe output trigger stall pending.
#define LWE2B5_RAISE_BUFFER_0                      (0x26)
// 5 bit raise vector
#define LWE2B5_RAISE_BUFFER_0_RAISE_VECTOR_BUFFER                   4:0

// Raise buffer control during V Blank.
//  This specifies when to return raise buffer
//  if it is issued during vertical blank time.
#define LWE2B5_RAISE_BUFFER_0_RAISE_BUFFER_VBLANK                   15:15
#define LWE2B5_RAISE_BUFFER_0_RAISE_BUFFER_VBLANK_IMMEDIATE                        (0)    // // Raise buffer is acknowledged immediately if
//  it is issued during vertical blank time.

#define LWE2B5_RAISE_BUFFER_0_RAISE_BUFFER_VBLANK_DELAYED                  (1)    // // Raise buffer is acknowledged at the end of
//  the first buffer of the next frame if it
//  is issued during vertical blank time.


// 4 bit channel ID which will be returned
//  when the end-of-buffer event oclwrs after
//  raise buffer is issued.
#define LWE2B5_RAISE_BUFFER_0_CHANNEL_ID_BUFFER                     19:16


// The end of frame event is generated when
//  the last data of the output buffer is
//  written to memory. This is independent of
//  output trigger so the end of buffer raise
//  acknowledge is returned even though there
//  maybe output trigger stall pending.
// If this raise is issued when
#define LWE2B5_RAISE_FRAME_0                       (0x27)
// 5 bit raise vector
#define LWE2B5_RAISE_FRAME_0_RAISE_VECTOR_FRAME                     4:0

// Raise frame control during V Blank.
//  This specifies when to return raise frame
//  if it is issued during vertical blank time.
#define LWE2B5_RAISE_FRAME_0_RAISE_FRAME_VBLANK                     15:15
#define LWE2B5_RAISE_FRAME_0_RAISE_FRAME_VBLANK_IMMEDIATE                  (0)    // // Raise frame is acknowledged immediately if
//  it is issued during vertical blank time.

#define LWE2B5_RAISE_FRAME_0_RAISE_FRAME_VBLANK_DELAYED                    (1)    // // Raise frame is acknowledged at the end of
//  the next frame if it is issued during
//  vertical blank time.


// 4 bit channel ID which will be returned
//  when the end-of-frame event oclwrs after
//  raise frame is issued.
#define LWE2B5_RAISE_FRAME_0_CHANNEL_ID_FRAME                       19:16

// Starting with SC17, REFCOUNT is increased from 16 bits to 32 bits.

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

#define LWE2B5_EPP_MCCIF_FIFOCTRL_0                        (0x29)
#define LWE2B5_EPP_MCCIF_FIFOCTRL_0_EPP_MCCIF_WRCL_MCLE2X                   0:0
#define LWE2B5_EPP_MCCIF_FIFOCTRL_0_EPP_MCCIF_WRCL_MCLE2X_INIT_ENUM                        DISABLE
#define LWE2B5_EPP_MCCIF_FIFOCTRL_0_EPP_MCCIF_WRCL_MCLE2X_DISABLE                  (0)
#define LWE2B5_EPP_MCCIF_FIFOCTRL_0_EPP_MCCIF_WRCL_MCLE2X_ENABLE                   (1)

#define LWE2B5_EPP_MCCIF_FIFOCTRL_0_EPP_MCCIF_RDMC_RDFAST                   1:1
#define LWE2B5_EPP_MCCIF_FIFOCTRL_0_EPP_MCCIF_RDMC_RDFAST_INIT_ENUM                        DISABLE
#define LWE2B5_EPP_MCCIF_FIFOCTRL_0_EPP_MCCIF_RDMC_RDFAST_DISABLE                  (0)
#define LWE2B5_EPP_MCCIF_FIFOCTRL_0_EPP_MCCIF_RDMC_RDFAST_ENABLE                   (1)

#define LWE2B5_EPP_MCCIF_FIFOCTRL_0_EPP_MCCIF_WRMC_CLLE2X                   2:2
#define LWE2B5_EPP_MCCIF_FIFOCTRL_0_EPP_MCCIF_WRMC_CLLE2X_INIT_ENUM                        DISABLE
#define LWE2B5_EPP_MCCIF_FIFOCTRL_0_EPP_MCCIF_WRMC_CLLE2X_DISABLE                  (0)
#define LWE2B5_EPP_MCCIF_FIFOCTRL_0_EPP_MCCIF_WRMC_CLLE2X_ENABLE                   (1)

#define LWE2B5_EPP_MCCIF_FIFOCTRL_0_EPP_MCCIF_RDCL_RDFAST                   3:3
#define LWE2B5_EPP_MCCIF_FIFOCTRL_0_EPP_MCCIF_RDCL_RDFAST_INIT_ENUM                        DISABLE
#define LWE2B5_EPP_MCCIF_FIFOCTRL_0_EPP_MCCIF_RDCL_RDFAST_DISABLE                  (0)
#define LWE2B5_EPP_MCCIF_FIFOCTRL_0_EPP_MCCIF_RDCL_RDFAST_ENABLE                   (1)

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

#define LWE2B5_TIMEOUT_WCOAL_EPP_0                 (0x2a)
#define LWE2B5_TIMEOUT_WCOAL_EPP_0_EPPU_WCOAL_TMVAL                 7:0

#define LWE2B5_TIMEOUT_WCOAL_EPP_0_EPPV_WCOAL_TMVAL                 15:8

#define LWE2B5_TIMEOUT_WCOAL_EPP_0_EPPY_WCOAL_TMVAL                 23:16

// Memory Client High-Priority Control Register
// This register exists only for clients with high-priority. Reset values are 0 (disabled).
// The high-priority should be enabled for hard real-time clients only. The values to program
// depend on the client bandwidth requirement and the client versus memory controllers clolck ratio.
// The high-priority is set if the number of entries in the return data fifo is under the threshold.
// The high-priority assertion can be delayed by a number of memory clock cycles indicated by the timer.
// This creates an hysteresis effect, avoiding setting the high-priority for very short periods of time,
// which may or may not be desirable.

#define LWE2B5_MCCIF_EPPUP_HP_0                    (0x2b)
#define LWE2B5_MCCIF_EPPUP_HP_0_CBR_EPPUP2MC_HPTH                   3:0

#define LWE2B5_MCCIF_EPPUP_HP_0_CBR_EPPUP2MC_HPTM                   21:16

// Memory Client High-Priority Control Register
// This register exists only for clients with high-priority. Reset values are 0 (disabled).
// The high-priority should be enabled for hard real-time clients only. The values to program
// depend on the client bandwidth requirement and the client versus memory controllers clolck ratio.
// The high-priority is set if the number of entries in the data fifo is higher than the threshold.

#define LWE2B5_MCCIF_EPPU_HP_0                     (0x2c)
#define LWE2B5_MCCIF_EPPU_HP_0_CBW_EPPU2MC_HPTH                     6:0

// Memory Client High-Priority Control Register
// This register exists only for clients with high-priority. Reset values are 0 (disabled).
// The high-priority should be enabled for hard real-time clients only. The values to program
// depend on the client bandwidth requirement and the client versus memory controllers clolck ratio.
// The high-priority is set if the number of entries in the data fifo is higher than the threshold.

#define LWE2B5_MCCIF_EPPV_HP_0                     (0x2d)
#define LWE2B5_MCCIF_EPPV_HP_0_CBW_EPPV2MC_HPTH                     6:0

// Memory Client High-Priority Control Register
// This register exists only for clients with high-priority. Reset values are 0 (disabled).
// The high-priority should be enabled for hard real-time clients only. The values to program
// depend on the client bandwidth requirement and the client versus memory controllers clolck ratio.
// The high-priority is set if the number of entries in the data fifo is higher than the threshold.

#define LWE2B5_MCCIF_EPPY_HP_0                     (0x2e)
#define LWE2B5_MCCIF_EPPY_HP_0_CBW_EPPY2MC_HPTH                     6:0

// Memory Client Hysteresis Control Register
// This register exists only for clients with hysteresis.
// BUG 505006: Hysteresis configuration can only be updated when memory traffic is idle.
// HYST_EN can be used to turn on or off the hysteresis logic.
// HYST_REQ_TH is the threshold of pending requests required
//   before allowing them to pass through
//   (overriden after HYST_REQ_TM cycles).
// Hysteresis logic will stop holding request after (1<<HYST_TM) cycles
//   (this should not have to be used and is only a WAR for
//   unexpected hangs).
// Deep hysteresis is a second level of hysteresis on a longer time-frame.
//   DHYST_TH is the size of the read burst (requests are held until there
//   is space for the entire burst in the return data fifo).
//   During a burst period, if there are no new requests after
//   DHYST_TM cycles, then the burst is terminated early.

#define LWE2B5_MCCIF_EPPUP_HYST_0                  (0x2f)
#define LWE2B5_MCCIF_EPPUP_HYST_0_CBR_EPPUP2MC_HYST_REQ_TM                  7:0

#define LWE2B5_MCCIF_EPPUP_HYST_0_CBR_EPPUP2MC_DHYST_TM                     15:8

#define LWE2B5_MCCIF_EPPUP_HYST_0_CBR_EPPUP2MC_DHYST_TH                     23:16

#define LWE2B5_MCCIF_EPPUP_HYST_0_CBR_EPPUP2MC_HYST_TM                      27:24

#define LWE2B5_MCCIF_EPPUP_HYST_0_CBR_EPPUP2MC_HYST_REQ_TH                  30:28

#define LWE2B5_MCCIF_EPPUP_HYST_0_CBR_EPPUP2MC_HYST_EN                      31:31
#define LWE2B5_MCCIF_EPPUP_HYST_0_CBR_EPPUP2MC_HYST_EN_INIT_ENUM                   ENABLE
#define LWE2B5_MCCIF_EPPUP_HYST_0_CBR_EPPUP2MC_HYST_EN_ENABLE                      (1)
#define LWE2B5_MCCIF_EPPUP_HYST_0_CBR_EPPUP2MC_HYST_EN_DISABLE                     (0)

// Memory Client Hysteresis Control Register
// This register exists only for clients with hysteresis.
// BUG 505006: Hysteresis configuration can only be updated when memory traffic is idle.
// HYST_EN can be used to turn on or off the hysteresis logic.
// HYST_REQ_TH is the threshold of pending requests required
//   before allowing them to pass through
//   (overriden after HYST_REQ_TM cycles).

#define LWE2B5_MCCIF_EPPU_HYST_0                   (0x30)
#define LWE2B5_MCCIF_EPPU_HYST_0_CBW_EPPU2MC_HYST_REQ_TM                    11:0

#define LWE2B5_MCCIF_EPPU_HYST_0_CBW_EPPU2MC_HYST_REQ_TH                    30:28

#define LWE2B5_MCCIF_EPPU_HYST_0_CBW_EPPU2MC_HYST_EN                31:31
#define LWE2B5_MCCIF_EPPU_HYST_0_CBW_EPPU2MC_HYST_EN_INIT_ENUM                     ENABLE
#define LWE2B5_MCCIF_EPPU_HYST_0_CBW_EPPU2MC_HYST_EN_ENABLE                        (1)
#define LWE2B5_MCCIF_EPPU_HYST_0_CBW_EPPU2MC_HYST_EN_DISABLE                       (0)

// Memory Client Hysteresis Control Register
// This register exists only for clients with hysteresis.
// BUG 505006: Hysteresis configuration can only be updated when memory traffic is idle.
// HYST_EN can be used to turn on or off the hysteresis logic.
// HYST_REQ_TH is the threshold of pending requests required
//   before allowing them to pass through
//   (overriden after HYST_REQ_TM cycles).

#define LWE2B5_MCCIF_EPPV_HYST_0                   (0x31)
#define LWE2B5_MCCIF_EPPV_HYST_0_CBW_EPPV2MC_HYST_REQ_TM                    11:0

#define LWE2B5_MCCIF_EPPV_HYST_0_CBW_EPPV2MC_HYST_REQ_TH                    30:28

#define LWE2B5_MCCIF_EPPV_HYST_0_CBW_EPPV2MC_HYST_EN                31:31
#define LWE2B5_MCCIF_EPPV_HYST_0_CBW_EPPV2MC_HYST_EN_INIT_ENUM                     ENABLE
#define LWE2B5_MCCIF_EPPV_HYST_0_CBW_EPPV2MC_HYST_EN_ENABLE                        (1)
#define LWE2B5_MCCIF_EPPV_HYST_0_CBW_EPPV2MC_HYST_EN_DISABLE                       (0)

// Memory Client Hysteresis Control Register
// This register exists only for clients with hysteresis.
// BUG 505006: Hysteresis configuration can only be updated when memory traffic is idle.
// HYST_EN can be used to turn on or off the hysteresis logic.
// HYST_REQ_TH is the threshold of pending requests required
//   before allowing them to pass through
//   (overriden after HYST_REQ_TM cycles).

#define LWE2B5_MCCIF_EPPY_HYST_0                   (0x32)
#define LWE2B5_MCCIF_EPPY_HYST_0_CBW_EPPY2MC_HYST_REQ_TM                    11:0

#define LWE2B5_MCCIF_EPPY_HYST_0_CBW_EPPY2MC_HYST_REQ_TH                    30:28

#define LWE2B5_MCCIF_EPPY_HYST_0_CBW_EPPY2MC_HYST_EN                31:31
#define LWE2B5_MCCIF_EPPY_HYST_0_CBW_EPPY2MC_HYST_EN_INIT_ENUM                     ENABLE
#define LWE2B5_MCCIF_EPPY_HYST_0_CBW_EPPY2MC_HYST_EN_ENABLE                        (1)
#define LWE2B5_MCCIF_EPPY_HYST_0_CBW_EPPY2MC_HYST_EN_DISABLE                       (0)

// In SC15/14, memory client register oclwpies index 0x1C (hex).
// Align address to index 0x3f for adding additional registers.
// This reserve 35 registers for expansion of MC regs in case it is needed in the future.
// Second-level clock enable override register
//
// This can override the 2nd level clock enables in case of malfunction.
// Only exposed to software when needed.
//
// Reminder for Sharath: Please determine correct clock names and add necessary bits here.
//

#define LWE2B5_CLKEN_OVERRIDE_0                    (0x40)
// <clk1> gated clock override.
#define LWE2B5_CLKEN_OVERRIDE_0_epp2clk_eppbm_clken_ov                      0:0
#define LWE2B5_CLKEN_OVERRIDE_0_epp2clk_eppbm_clken_ov_CLK_GATED                   (0)    // // Clock gating enabled.

#define LWE2B5_CLKEN_OVERRIDE_0_epp2clk_eppbm_clken_ov_CLK_ALWAYS_ON                       (1)    // // Clock gating disabled - clock is always on.


// <clk2> gated clock override.
#define LWE2B5_CLKEN_OVERRIDE_0_epp2clk_eppdp_clken_ov                      1:1
#define LWE2B5_CLKEN_OVERRIDE_0_epp2clk_eppdp_clken_ov_CLK_GATED                   (0)    // // Clock gating enabled.

#define LWE2B5_CLKEN_OVERRIDE_0_epp2clk_eppdp_clken_ov_CLK_ALWAYS_ON                       (1)    // // Clock gating disabled - clock is always on.



// Setting bits in this register enable
//  the corresponding interrrupt event to
//  generate a pending interrupt. Interrupt
//  output signal will be activated only if
//  the corresponding interrupt is not masked.
// Disabling an interrupt will not clear
//  a corresponding pending interrupt - it
//  only prevent a new interrupt event to
//  generate a pending interrupt.
#define LWE2B5_INTMASK_0                   (0x41)
// Context Switch Interrupt Mask.
// If the auto ack bit in the CTXSW register
//  is disabled, then software must enable
//  interrupt and unmask interrupt mask
//  to get context switch acknowledge
#define LWE2B5_INTMASK_0_CTXSW_INT_MASK                     0:0
#define LWE2B5_INTMASK_0_CTXSW_INT_MASK_MASKED                     (0)    // // Interrupt masked (disabled).

#define LWE2B5_INTMASK_0_CTXSW_INT_MASK_NOTMASKED                  (1)    // // Interrupt not masked (enabled). Note that
//   this interrupt is enabled upon reset.


// Frame End Interrupt Mask.
#define LWE2B5_INTMASK_0_FRAME_END_INT_MASK                 1:1
#define LWE2B5_INTMASK_0_FRAME_END_INT_MASK_MASKED                 (0)    // // Interrupt masked (disabled).

#define LWE2B5_INTMASK_0_FRAME_END_INT_MASK_NOTMASKED                      (1)    // // Interrupt not masked (enabled).


// Output Buffer End Interrupt Mask.
#define LWE2B5_INTMASK_0_BUFFER_END_INT_MASK                2:2
#define LWE2B5_INTMASK_0_BUFFER_END_INT_MASK_MASKED                        (0)    // // Interrupt masked (disabled).

#define LWE2B5_INTMASK_0_BUFFER_END_INT_MASK_NOTMASKED                     (1)    // // Interrupt not masked (enabled).


// Short Frmae Interrupt Mask.
#define LWE2B5_INTMASK_0_SHORT_FRAME_INT_MASK                       8:8
#define LWE2B5_INTMASK_0_SHORT_FRAME_INT_MASK_MASKED                       (0)    // // Interrupt masked (disabled).

#define LWE2B5_INTMASK_0_SHORT_FRAME_INT_MASK_NOTMASKED                    (1)    // // Interrupt not masked (enabled).



//  Writing to this register causes the internal
//  output buffer index counter to be reloaded
//  with OB_INDEX write data.
//  This register must not be written when
//  EPP is in the middle of buffer processing.
//  When EPP is disabled, internal output buffer
//  index is reset to zero.
#define LWE2B5_START_OB_INDEX_0                    (0x42)
// Output buffer starting index. Valid value
//  is 0 to (OB0_COUNT-1).
#define LWE2B5_START_OB_INDEX_0_START_OB_INDEX                      7:0


#define LWE2B5_EPP_DEBUG_FRAME_STATUS_REGISTER_0                   (0x43)
// Count's the number of SOF's received.
#define LWE2B5_EPP_DEBUG_FRAME_STATUS_REGISTER_0_FRAME_COUNT_STATUS                 15:0


#define LWE2B5_EPP_DEBUG_LINE_STATUS_REGISTER_0                    (0x44)
// The height count of the frame written to MC
#define LWE2B5_EPP_DEBUG_LINE_STATUS_REGISTER_0_LINE_COUNT_STATUS                   15:0


#define LWE2B5_EPP_DEBUG_FLOW_CONTROL_COUNTER_0                    (0x45)
#define LWE2B5_EPP_DEBUG_FLOW_CONTROL_COUNTER_0_BUFFER_COUNT                7:0


#define LWE2B5_BUFFER_RELEASE_0                    (0x46)
#define LWE2B5_BUFFER_RELEASE_0_BUFFER_RELEASE                      0:0


#define LWE2B5_OB0_BUFFER_ADDR_MODE_0                      (0x47)
#define LWE2B5_OB0_BUFFER_ADDR_MODE_0_Y_TILE_MODE                   0:0
#define LWE2B5_OB0_BUFFER_ADDR_MODE_0_Y_TILE_MODE_LINEAR                   (0)
#define LWE2B5_OB0_BUFFER_ADDR_MODE_0_Y_TILE_MODE_TILED                    (1)

#define LWE2B5_OB0_BUFFER_ADDR_MODE_0_UV_TILE_MODE                  16:16
#define LWE2B5_OB0_BUFFER_ADDR_MODE_0_UV_TILE_MODE_LINEAR                  (0)
#define LWE2B5_OB0_BUFFER_ADDR_MODE_0_UV_TILE_MODE_TILED                   (1)


#define LWE2B5_RESERVE_0_0                 (0x48)
#define LWE2B5_RESERVE_0_0_nc_RESERVE_0_0                   3:0

#define LWE2B5_RESERVE_0_0_nc_RESERVE_0_1                   7:4

#define LWE2B5_RESERVE_0_0_nc_RESERVE_0_2                   11:8

#define LWE2B5_RESERVE_0_0_nc_RESERVE_0_3                   15:12


#define LWE2B5_RESERVE_1_0                 (0x49)
#define LWE2B5_RESERVE_1_0_nc_RESERVE_1_0                   3:0

#define LWE2B5_RESERVE_1_0_nc_RESERVE_1_1                   7:4

#define LWE2B5_RESERVE_1_0_nc_RESERVE_1_2                   11:8

#define LWE2B5_RESERVE_1_0_nc_RESERVE_1_3                   15:12


#define LWE2B5_RESERVE_2_0                 (0x4a)
#define LWE2B5_RESERVE_2_0_nc_RESERVE_2_0                   3:0

#define LWE2B5_RESERVE_2_0_nc_RESERVE_2_1                   7:4

#define LWE2B5_RESERVE_2_0_nc_RESERVE_2_2                   11:8

#define LWE2B5_RESERVE_2_0_nc_RESERVE_2_3                   15:12


#define LWE2B5_RESERVE_3_0                 (0x4b)
#define LWE2B5_RESERVE_3_0_nc_RESERVE_3_0                   3:0

#define LWE2B5_RESERVE_3_0_nc_RESERVE_3_1                   7:4

#define LWE2B5_RESERVE_3_0_nc_RESERVE_3_2                   11:8

#define LWE2B5_RESERVE_3_0_nc_RESERVE_3_3                   15:12


#define LWE2B5_RESERVE_4_0                 (0x4c)
#define LWE2B5_RESERVE_4_0_nc_RESERVE_4_0                   3:0

#define LWE2B5_RESERVE_4_0_nc_RESERVE_4_1                   7:4

#define LWE2B5_RESERVE_4_0_nc_RESERVE_4_2                   11:8

#define LWE2B5_RESERVE_4_0_nc_RESERVE_4_3                   15:12

// Align address to 0x200 for second context registers

//
// REGISTER LIST
//
#define LIST_AREPP_REGS(_op_) \
_op_(LWE2B5_INCR_SYNCPT_0) \
_op_(LWE2B5_INCR_SYNCPT_CNTRL_0) \
_op_(LWE2B5_INCR_SYNCPT_ERROR_0) \
_op_(LWE2B5_EPP_SYNCPT_DEST_0) \
_op_(LWE2B5_CTXSW_0) \
_op_(LWE2B5_INTSTATUS_0) \
_op_(LWE2B5_EPP_CONTROL_0) \
_op_(LWE2B5_OUTPUT_FRAME_SIZE_0) \
_op_(LWE2B5_INPUT_FRAME_AOI_0) \
_op_(LWE2B5_OUTPUT_SCAN_DIR_0) \
_op_(LWE2B5_OB0_START_ADDRESS_Y_0) \
_op_(LWE2B5_OB0_BASE_ADDRESS_Y_0) \
_op_(LWE2B5_OB0_START_ADDRESS_U_0) \
_op_(LWE2B5_OB0_BASE_ADDRESS_U_0) \
_op_(LWE2B5_OB0_START_ADDRESS_V_0) \
_op_(LWE2B5_OB0_BASE_ADDRESS_V_0) \
_op_(LWE2B5_OB0_XY_OFFSET_LUMA_0) \
_op_(LWE2B5_OB0_XY_OFFSET_CHROMA_0) \
_op_(LWE2B5_OB0_SIZE_0) \
_op_(LWE2B5_OB0_LINE_STRIDE_L_0) \
_op_(LWE2B5_OB0_BUFFER_STRIDE_LUMA_0) \
_op_(LWE2B5_OB0_BUFFER_STRIDE_CHROMA_0) \
_op_(LWE2B5_CSC_RGB2Y_COEFF_0) \
_op_(LWE2B5_CSC_RGB2U_COEFF_0) \
_op_(LWE2B5_CSC_RGB2V_COEFF_0) \
_op_(LWE2B5_CSC_YOFFSET_COEFF_0) \
_op_(LWE2B5_FILTER_BOUND_0) \
_op_(LWE2B5_FILTER_BASE_0) \
_op_(LWE2B5_PP_FILTER_COEF_0) \
_op_(LWE2B5_CHROMA_FILTER_COEF_0) \
_op_(LWE2B5_ALFA_VALUE_0) \
_op_(LWE2B5_U_LINE_BUFFER_ADDR_0) \
_op_(LWE2B5_V_LINE_BUFFER_ADDR_0) \
_op_(LWE2B5_RAISE_BUFFER_0) \
_op_(LWE2B5_RAISE_FRAME_0) \
_op_(LWE2B5_EPP_MCCIF_FIFOCTRL_0) \
_op_(LWE2B5_TIMEOUT_WCOAL_EPP_0) \
_op_(LWE2B5_MCCIF_EPPUP_HP_0) \
_op_(LWE2B5_MCCIF_EPPU_HP_0) \
_op_(LWE2B5_MCCIF_EPPV_HP_0) \
_op_(LWE2B5_MCCIF_EPPY_HP_0) \
_op_(LWE2B5_MCCIF_EPPUP_HYST_0) \
_op_(LWE2B5_MCCIF_EPPU_HYST_0) \
_op_(LWE2B5_MCCIF_EPPV_HYST_0) \
_op_(LWE2B5_MCCIF_EPPY_HYST_0) \
_op_(LWE2B5_CLKEN_OVERRIDE_0) \
_op_(LWE2B5_INTMASK_0) \
_op_(LWE2B5_START_OB_INDEX_0) \
_op_(LWE2B5_EPP_DEBUG_FRAME_STATUS_REGISTER_0) \
_op_(LWE2B5_EPP_DEBUG_LINE_STATUS_REGISTER_0) \
_op_(LWE2B5_EPP_DEBUG_FLOW_CONTROL_COUNTER_0) \
_op_(LWE2B5_BUFFER_RELEASE_0) \
_op_(LWE2B5_OB0_BUFFER_ADDR_MODE_0) \
_op_(LWE2B5_RESERVE_0_0) \
_op_(LWE2B5_RESERVE_1_0) \
_op_(LWE2B5_RESERVE_2_0) \
_op_(LWE2B5_RESERVE_3_0) \
_op_(LWE2B5_RESERVE_4_0)

//
// ADDRESS SPACES
//
#define BASE_ADDRESS_EPP        0x00000000

//
// AREPP REGISTER BANKS
//
#define EPP0_FIRST_REG 0x0000   // EPP_INCR_SYNCPT_0
#define EPP0_LAST_REG 0x0002    // EPP_INCR_SYNCPT_ERROR_0
#define EPP1_FIRST_REG 0x0008   // EPP_EPP_SYNCPT_DEST_0
#define EPP1_LAST_REG 0x0027    // EPP_RAISE_FRAME_0
#define EPP2_FIRST_REG 0x0029   // EPP_EPP_MCCIF_FIFOCTRL_0
#define EPP2_LAST_REG 0x0032    // EPP_MCCIF_EPPY_HYST_0
#define EPP3_FIRST_REG 0x0040   // EPP_CLKEN_OVERRIDE_0
#define EPP3_LAST_REG 0x004c    // EPP_RESERVE_4_0

#endif // ifndef _cl_e2b5_h_


