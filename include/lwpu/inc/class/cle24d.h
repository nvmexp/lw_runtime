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
#ifndef _cl_e24d_h_
#define _cl_e24d_h_

#include "lwtypes.h"
#define LWE2_CAPTURE                   (0xE24D)

/*
 * Video Camera Interface register definition
 *
 * The Video Camera Interface takes input data from the VI port or from host.
 * Data from VI port can be in the following format:
 *  a. ITU-R BT.656: 1-byte/clock U8Y8V8Y8 format with embedded syncs in the data stream.
 *  b. YUV422: 1-byte/clock U8Y8V8Y8 format with H sync on VHS pin and V sync on VVS pin.
 *  c. Bayer Pattern (lwrrently not supported): R8G8, G8B8 format with H sync on VHS pin and
 *     V sync on VVS pin.
 *  For case b and c, it is also possible to generate H sync and V sync internally in VI
 *  module. These internally generated H sync and V sync can be output to the external device
 *  and also used internally by the VI module.
 * Data from Host can be in the following format:
 *  a. YUV422: non-planar 32-bit U8Y8V8Y8 format going through Y-FIFO.
 *  b. YUV420: planar 32-bit Y, U, V format going through Y-FIFO, U-FIFO, V-FIFO
 *     correspondingly.
 * **** In the future, data from host should come from command buffer interface where YUV420
 *      to YUV422 colwersion, if necessary, should be done using the command buffer.
 *      It may not be necessary to colwert YUV420 to YUV422 if there is no image processing
 *      needed.
 *
 * The processing stages are:
 *  a. Horizontal low-pass filtering
 *  b. Horizontal down-scaling with or without horizontal averaging
 *  c. Vertical down-scaling with or without vertical averaging
 *  d. YUV to RGB Color Space Colwersion
 *
 * Output can be sent to memory (typically for previewing the video image on the display) and/or
 * can be sent to Encoder Pre-Processor module to be encoded.
 *
 * Interface to memory is a normal Block Write or a YUV Block Write interface with option
 * for horizontal flip, vertical flip, and XY transpose. Data to be stored in memory can be
 * in 16-bit RGB format with optional dithering, YUV422 non-planar, and YUV422/420 planar.
 * If output data is stored as YUV420 planar format, the chroma data averaging can be optionally
 * performed for each pair of input lines.
 * Normal block write is used when output data is in RGB or YUV non-planar format.
 * YUV block write is used when output data is YUV planar format.
 *
 * Output data stored in memory is stored in one buffer set: video buffer 0.
 * Output buffer set 0 consists of a programmable number of buffers (from 1 to 255) defined by
 * VB0_COUNT parameter.
 * Each output buffer consists of programmable number of lines (max 1 frame) defined by
 * VB0_V_SIZE parameter which should be even number when data is stored in YUV420 format.
 * VB0_H_SIZE parameter determines the line stride (in pixels) and VB0_BUFFER_STRIDE
 * determines the buffer stride. The number of active pixels per line in the output buffer
 * depends on the input video horizontal active period and the scaling factor and should
 * typically not more than the line stride.
 * When output DMA request is enabled, the encoder will send a request to send the output
 * buffers to the host. Request will be sent after each output buffer is filled and also at the
 * end of each encoded frame.
 * Note that at the end of frame, the encoded data may not end at output buffer boundary.
 * The encoder will send the start address of the buffer to be transferred via the output DMA
 * and the correct size of the buffer with each request.
 * These are 32-bit registers that can be written/read from host register interface.
 * AP15 flow control
 * For AP15, we will not use host1xRDMA engine. VI will write into output buffer as is. SW will
 * use SYNCPT_OP_DONE as an indicator of one VI output buffer is ready for read. After SW
 * consumes one buffer, SW will write to BUFFER_RELEASE register.
 * EPP has an internal counter. Every buffer filed will increament the buffer, every write from
 * SW to BUFFER_RELEASE register will decrease this counter. VI will stall input bus if:
 * counter >= EPP_OUTPUT_BUFFER_COUNT - 1
 * For SW to use this flow control correctly, SW has to release all the buffers that locked by
 * VI to maintain synchronizations of SW and VI. For example, after flow control is enabled,
 * VI output 4 buffers, our flowControlBufferCount = 4. SW only need 2 of them. SW should
 * write to BUFFER_RELEASE register 4 times before switch VI for other stream capturing. There
 * is no reset of this counter or wrap around. This buffer will be zero after reset. VI
 * RTL does provide a EPP_DEBUG_CONTROL_FLOW_COUNTER register, but it is for debug only.
 * This apply to both Output1 and Output2.
 * suggested syncpt programming sequence: mail from sep.7th, 2007
 * -ISP single shot is definitely broken.  The current ECO is probably not correct.
 * When VI receives EOF from ISP (in single shot mode), we should squash subsequent
 * vsyncs but NOT hsyncs or data.
 * -We should test what happens if ISP gets too many lines in a frame
 * We think that the following sequence will work in all cases...
 * enable continuous vi op_done
 * while(1) {
 *      program pipe
 *      program stream defines for entire pipe
 *      ilwoke single shot
 *      issue start_write
 *      wait start_write
 *      issue reg_wr_safe
 *  flush buffer
 *      wait op_done
 *  trigger next unit
 *  wait reg_write_safe
 * }
 * syncpt commentes:
 *
 * VI has two different types of syncpt, single-shot and continuous.
 * Single-shot syncpts are requested by SW via a write to the one of the INCR_SYNCPT registers.
 * When the condition becomes true, the syncpt is returned.  Continuous syncpts are enabled by
 * a write to a CONT_SYNCPT register, and will be returned whenever the condition becomes true,
 * and does not require SW to do an INCR_SYNCPT write.
 *
 * single-shot synpct: There are three registers related to single-shot syncpt,
 *   VI_OUT_1_INCR_SYNCPT - applies to VI Memory Channel 1
 *   VI_OUT_2_INCR_SYNCPT - applies to VI Memory Channel 2
 *   VI_MISC_INCR_SYNPCT  - applies to non-memory related conditions
 *
 * condition: There are 5 conditions for VI_OUT_1_INCR_SYNCPT and VI_OUT_2_INCR_SYNCPT
 *   0 -- immediate : syncpt index will be returned immediately when VI_OUT_1/2_INCR_SYNCPT is written.
 *   1 -- OP_DONE: syncpt index will be returned when the corresponding output is idle, either output1
 *        or output2. This syncpt is level triggered.
 *   2 -- RD_DONE:  this is treated the same as OP_DONE condition
 *   3 -- REG_WR_SAFE: when all the resources defined in RESOURCE_DEFINE register are all idle, the
 *        syncpt index will be returned. This synpct is level triggered.
 *   4 -- START_WRITE: when the first pixel is written to memory, either from camera or host, the syncpt
 *        index will be returned. This syncpt is edge triggered.
 * condition: There are 9 conditions for VI_MISC_INCR_SYNCPT
 *            single-shot syncpts (excpet immediate) are not supported on the MISC syncpt.
 *   0 -- immediate: syncpt index will be returned immediately when VI_MISC_INCR_SYNCPT is written.
 *        This is a level triggered syncpt.
 *
 * continuous syncpt: There are eight continuous syncpt in VI. Each of them can be enable by set the
 * ENABLE bit along with syncpt index field.
 * Whenever a continous syncpt is enabled, the corresponding single-shot syncpt may not be used.
 *
 * VI_CONT_SYNCPT_OUT_1:
 *      Condition for syncpt return is OP_DONE from output 1.
 * VI_CONT_SYNCPT_OUT_2:
 *      Condition for syncpt return is OP_DONE from output 1.
 * VI_CONT_SYNCPT_VIP_VSYNC:
 *      syncpt index will be returned when the first vsync from VIP input is received. This is an
 *      edge triggered syncpt.
 * VI_CONT_SYNCPT_VI2EPP:
 *      This condition will forward syncpt to EPP whenever data is sent to EPP. It will forward once per
 *      EPP buffer. The syncpt will sent to EPP at the first line of the buffer, and after every
 *      LINES_PER_BUFFER lines (as defined by the EPP_LINES_PER_BUFFER register). EPP will return the syncpt
 *      when the last byte of a buffer is written into memory(tag returned).
 * VI_CONT_SYNCPT_CSI_PPA_FRAME_START:
 *      The condition for this syncpt is CSI PPA port received a frame start.
 * VI_CONT_SYNCPT_CSI_PPA_FRAME_END:
 *      The condition for this syncpt is CSI PPA port received a frame end.
 * VI_CONT_SYNCPT_CSI_PPB_FRAME_START:
 *      The condition for this syncpt is CSI PPB port received a frame start. MISC_CSI_PPB_FRAME_START.
 * VI_CONT_SYNCPT_CSI_PPB_FRAME_END:
 *      The condition for this syncpt is CSI PPB port received a frame end.
 *
 * REG_WR_SAFE's "safe" condition is defined by the VI_STREAM_1_RESOURCE_DEFINE (for OUT_1) and
 * VI_STREAM_2_RESOURCE_DEFINE (for OUT_2).  The syncpt will return when all the requested resources are IDLE.
 * If no resources are requested, it will return immediately.
 *
 * Since REG_WR_SAFE is level triggered, it should be used in conjuction with START_WRITE.  In the format of:
 *      INCR_SYNCPT <START_WRITE>
 *      WAIT (START_WRITE)
 *      INCR_SYNCPT <REG_WR_SAFE>
 *
 * Continuous syncpt always use OP_DONE as condition. The mapping of continuous syncpt to single-shot syncpt:
 *   VI_CONT_SYNCPT_OUT_1                              mapped to VI_OUT_1_INCR_SYNCPT condition OP_DONE
 *   VI_CONT_SYNCPT_OUT_2                              mapped to VI_OUT_2_INCR_SYNCPT condition OP_DONE
 *   VI_CONT_SYNCPT_VIP_VSYN                        mapped to VI_MISC_INCR_SYNCPT condition MISC_VIP_VSYNC
 *   VI_CONT_SYNCPT_CSI_PPA_FRAME_START mapped to VI_MISC_INCR_SYNCPT condition MISC_CSI_PPA_FRAME_START
 *   VI_CONT_SYNCPT_CSI_PPA_FRAME_END     mapped to VI_MISC_INCR_SYNCPT condition MISC_CSI_PPA_FRAME_END
 *   VI_CONT_SYNCPT_CSI_PPB_FRAME_START mapped to VI_MISC_INCR_SYNCPT condition MISC_CSI_PPB_FRAME_START
 *   VI_CONT_SYNCPT_CSI_PPB_FRAME_END     mapped to VI_MISC_INCR_SYNCPT condition MISC_CSI_PPB_FRAME_END
 *   VI_CONT_SYNCPT_VI2EPP                             mapped to EPP_INCR_SYNCPT condition OP_DONE
 * Can not program continuous syncpt with mapping single_shot syncpt conditions. It is fine to program continuous
 * syncpt with other syncpt conditions. For example:
 *      enable VI_CONT_OUT_1 with VI_OUT_1_INCR_SYNCPT condition REG_WR_SAFE  -- ok
 *      enable VI_CONT_OUT_2 with VI_OUT_1_INCR_SYNCPT condition OP_DONE      -- ok
 *      enable VI_CONT_VIP_VSYNC with VI_MISC_INCR_SYNCPT condition VIP_VSYNC -- not ok
 *      enable VI_CONT_OUT1 with VI_OUT_1_INCR_SYNCPT condition OP_DONE       -- not ok
 * 
 *
 */

#define LWE24D_VI_OUT_1_INCR_SYNCPT_NB_CONDS                                    5

#define LWE24D_VI_OUT_1_INCR_SYNCPT_0                                          (0x00000000)

// Condition mapped from raise/wait
#define LWE24D_VI_OUT_1_INCR_SYNCPT_0_OUT_1_COND                               15:8
#define LWE24D_VI_OUT_1_INCR_SYNCPT_0_OUT_1_COND_IMMEDIATE                     (0)
#define LWE24D_VI_OUT_1_INCR_SYNCPT_0_OUT_1_COND_OP_DONE                       (1)
#define LWE24D_VI_OUT_1_INCR_SYNCPT_0_OUT_1_COND_RD_DONE                       (2)
#define LWE24D_VI_OUT_1_INCR_SYNCPT_0_OUT_1_COND_REG_WR_SAFE                   (3)
#define LWE24D_VI_OUT_1_INCR_SYNCPT_0_OUT_1_COND_START_WRITE                   (4)
#define LWE24D_VI_OUT_1_INCR_SYNCPT_0_OUT_1_COND_COND_5                        (5)
#define LWE24D_VI_OUT_1_INCR_SYNCPT_0_OUT_1_COND_COND_6                        (6)
#define LWE24D_VI_OUT_1_INCR_SYNCPT_0_OUT_1_COND_COND_7                        (7)
#define LWE24D_VI_OUT_1_INCR_SYNCPT_0_OUT_1_COND_COND_8                        (8)
#define LWE24D_VI_OUT_1_INCR_SYNCPT_0_OUT_1_COND_COND_9                        (9)
#define LWE24D_VI_OUT_1_INCR_SYNCPT_0_OUT_1_COND_COND_10                       (10)
#define LWE24D_VI_OUT_1_INCR_SYNCPT_0_OUT_1_COND_COND_11                       (11)
#define LWE24D_VI_OUT_1_INCR_SYNCPT_0_OUT_1_COND_COND_12                       (12)
#define LWE24D_VI_OUT_1_INCR_SYNCPT_0_OUT_1_COND_COND_13                       (13)
#define LWE24D_VI_OUT_1_INCR_SYNCPT_0_OUT_1_COND_COND_14                       (14)
#define LWE24D_VI_OUT_1_INCR_SYNCPT_0_OUT_1_COND_COND_15                       (15)

// syncpt index value
#define LWE24D_VI_OUT_1_INCR_SYNCPT_0_OUT_1_INDX                               7:0

#define LWE24D_VI_OUT_1_INCR_SYNCPT_CNTRL_0                                    (0x1)

// If NO_STALL is 1, then when fifos are full,
// INCR_SYNCPT methods will be dropped and the
// INCR_SYNCPT_ERROR[COND] bit will be set.
// If NO_STALL is 0, then when fifos are full,
// the client host interface will be stalled.
#define LWE24D_VI_OUT_1_INCR_SYNCPT_CNTRL_0_OUT_1_INCR_SYNCPT_NO_STALL         8:8

// If SOFT_RESET is set, then all internal state
// of the client syncpt block will be reset.
// To do soft reset, first set SOFT_RESET of
// all host1x clients affected, then clear all
// SOFT_RESETs.
#define LWE24D_VI_OUT_1_INCR_SYNCPT_CNTRL_0_OUT_1_INCR_SYNCPT_SOFT_RESET       0:0

#define LWE24D_VI_OUT_1_INCR_SYNCPT_ERROR_0                                    (0x2)

// COND_STATUS[COND] is set if the fifo for COND overflows.
// This bit is sticky and will remain set until cleared.
// Cleared by writing 1.
#define LWE24D_VI_OUT_1_INCR_SYNCPT_ERROR_0_OUT_1_COND_STATUS                  31:0

// just in case names were redefined using macros
#define LW_VI_OUT_2_INCR_SYNCPT_NB_CONDS        5

#define LWE24D_VI_OUT_2_INCR_SYNCPT_0                                          (0x8)
#define LWE24D_VI_OUT_2_INCR_SYNCPT_0_OUT_2_COND                               15:8
#define LWE24D_VI_OUT_2_INCR_SYNCPT_0_OUT_2_COND_IMMEDIATE                     (0)
#define LWE24D_VI_OUT_2_INCR_SYNCPT_0_OUT_2_COND_OP_DONE                       (1)
#define LWE24D_VI_OUT_2_INCR_SYNCPT_0_OUT_2_COND_RD_DONE                       (2)
#define LWE24D_VI_OUT_2_INCR_SYNCPT_0_OUT_2_COND_REG_WR_SAFE                   (3)
#define LWE24D_VI_OUT_2_INCR_SYNCPT_0_OUT_2_COND_START_WRITE                   (4)
#define LWE24D_VI_OUT_2_INCR_SYNCPT_0_OUT_2_COND_COND_5                        (5)
#define LWE24D_VI_OUT_2_INCR_SYNCPT_0_OUT_2_COND_COND_6                        (6)
#define LWE24D_VI_OUT_2_INCR_SYNCPT_0_OUT_2_COND_COND_7                        (7)
#define LWE24D_VI_OUT_2_INCR_SYNCPT_0_OUT_2_COND_COND_8                        (8)
#define LWE24D_VI_OUT_2_INCR_SYNCPT_0_OUT_2_COND_COND_9                        (9)
#define LWE24D_VI_OUT_2_INCR_SYNCPT_0_OUT_2_COND_COND_10                       (10)
#define LWE24D_VI_OUT_2_INCR_SYNCPT_0_OUT_2_COND_COND_11                       (11)
#define LWE24D_VI_OUT_2_INCR_SYNCPT_0_OUT_2_COND_COND_12                       (12)
#define LWE24D_VI_OUT_2_INCR_SYNCPT_0_OUT_2_COND_COND_13                       (13)
#define LWE24D_VI_OUT_2_INCR_SYNCPT_0_OUT_2_COND_COND_14                       (14)
#define LWE24D_VI_OUT_2_INCR_SYNCPT_0_OUT_2_COND_COND_15                       (15)
#define LWE24D_VI_OUT_2_INCR_SYNCPT_0_OUT_2_INDX                               7:0

#define LWE24D_VI_OUT_2_INCR_SYNCPT_CNTRL_0                                    (0x9)

// If NO_STALL is 1, then when fifos are full,
// INCR_SYNCPT methods will be dropped and the
// INCR_SYNCPT_ERROR[COND] bit will be set.
// If NO_STALL is 0, then when fifos are full,
// the client host interface will be stalled.
#define LWE24D_VI_OUT_2_INCR_SYNCPT_CNTRL_0_OUT_2_INCR_SYNCPT_NO_STALL         8:8

// If SOFT_RESET is set, then all internal state
// of the client syncpt block will be reset.
// To do soft reset, first set SOFT_RESET of
// all host1x clients affected, then clear all
// SOFT_RESETs.
#define LWE24D_VI_OUT_2_INCR_SYNCPT_CNTRL_0_OUT_2_INCR_SYNCPT_SOFT_RESET       0:0

#define LWE24D_VI_OUT_2_INCR_SYNCPT_ERROR_0                                    (0xa)

// COND_STATUS[COND] is set if the fifo for COND overflows.
// This bit is sticky and will remain set until cleared.
// Cleared by writing 1.
#define LWE24D_VI_OUT_2_INCR_SYNCPT_ERROR_0_OUT_2_COND_STATUS                  31:0

// just in case names were redefined using macros
#define LW_VI_MISC_INCR_SYNCPT_NB_CONDS 9

#define LWE24D_VI_MISC_INCR_SYNCPT_0                                           (0x10)
// Condition mapped from raise/wait
#define LWE24D_VI_MISC_INCR_SYNCPT_0_MISC_COND                                 15:8
#define LWE24D_VI_MISC_INCR_SYNCPT_0_MISC_COND_IMMEDIATE                       (0)
#define LWE24D_VI_MISC_INCR_SYNCPT_0_MISC_COND_OP_DONE                         (1)
#define LWE24D_VI_MISC_INCR_SYNCPT_0_MISC_COND_RD_DONE                         (2)
#define LWE24D_VI_MISC_INCR_SYNCPT_0_MISC_COND_REG_WR_SAFE                     (3)
#define LWE24D_VI_MISC_INCR_SYNCPT_0_MISC_COND_MISC_VIP_VSYNC                  (4)
#define LWE24D_VI_MISC_INCR_SYNCPT_0_MISC_COND_MISC_CSI_PPA_FRAME_START        (5)
#define LWE24D_VI_MISC_INCR_SYNCPT_0_MISC_COND_MISC_CSI_PPA_FRAME_END          (6)
#define LWE24D_VI_MISC_INCR_SYNCPT_0_MISC_COND_MISC_CSI_PPB_FRAME_START        (7)
#define LWE24D_VI_MISC_INCR_SYNCPT_0_MISC_COND_MISC_CSI_PPB_FRAME_END          (8)
#define LWE24D_VI_MISC_INCR_SYNCPT_0_MISC_COND_COND_9                          (9)
#define LWE24D_VI_MISC_INCR_SYNCPT_0_MISC_COND_COND_10                         (10)
#define LWE24D_VI_MISC_INCR_SYNCPT_0_MISC_COND_COND_11                         (11)
#define LWE24D_VI_MISC_INCR_SYNCPT_0_MISC_COND_COND_12                         (12)
#define LWE24D_VI_MISC_INCR_SYNCPT_0_MISC_COND_COND_13                         (13)
#define LWE24D_VI_MISC_INCR_SYNCPT_0_MISC_COND_COND_14                         (14)
#define LWE24D_VI_MISC_INCR_SYNCPT_0_MISC_COND_COND_15                         (15)

// syncpt index value
#define LWE24D_VI_MISC_INCR_SYNCPT_0_MISC_INDX                                 7:0

#define LWE24D_VI_MISC_INCR_SYNCPT_CNTRL_0                                     (0x11)
// If NO_STALL is 1, then when fifos are full,
// INCR_SYNCPT methods will be dropped and the
// INCR_SYNCPT_ERROR[COND] bit will be set.
// If NO_STALL is 0, then when fifos are full,
// the client host interface will be stalled.
#define LWE24D_VI_MISC_INCR_SYNCPT_CNTRL_0_MISC_INCR_SYNCPT_NO_STALL           8:8

// If SOFT_RESET is set, then all internal state
// of the client syncpt block will be reset.
// To do soft reset, first set SOFT_RESET of
// all host1x clients affected, then clear all
// SOFT_RESETs.
#define LWE24D_VI_MISC_INCR_SYNCPT_CNTRL_0_MISC_INCR_SYNCPT_SOFT_RESET         0:0

#define LWE24D_VI_MISC_INCR_SYNCPT_ERROR_0                                     (0x12)
// COND_STATUS[COND] is set if the fifo for COND overflows.
// This bit is sticky and will remain set until cleared.
// Cleared by writing 1.
#define LWE24D_VI_MISC_INCR_SYNCPT_ERROR_0_MISC_COND_STATUS                    31:0

// just in case names were redefined using macros

#define LWE24D_VI_CONT_SYNCPT_OUT_1_0                                          (0x18)
// return INDX (set HOST_CLRD packet TYPE field to SYNCPT)
#define LWE24D_VI_CONT_SYNCPT_OUT_1_0_INDX_OUT_1                               7:0

// on host read bus every time OUT_1 condition is true and OUT_1_EN is set
#define LWE24D_VI_CONT_SYNCPT_OUT_1_0_EN_OUT_1                                 8:8

#define LWE24D_VI_CONT_SYNCPT_OUT_2_0                                          (0x19)
// return INDX (set HOST_CLRD packet TYPE field to SYNCPT)
#define LWE24D_VI_CONT_SYNCPT_OUT_2_0_INDX_OUT_2                               7:0

// on host read bus every time OUT_2 condition is true and OUT_2_EN is set
#define LWE24D_VI_CONT_SYNCPT_OUT_2_0_EN_OUT_2                                 8:8

#define LWE24D_VI_CONT_SYNCPT_VIP_VSYNC_0                                      (0x1a)
// return INDX (set HOST_CLRD packet TYPE field to SYNCPT)
#define LWE24D_VI_CONT_SYNCPT_VIP_VSYNC_0_INDX_VIP_VSYNC                       7:0

// on host read bus every time VSYNC condition is true and EN_VSYNC is set
#define LWE24D_VI_CONT_SYNCPT_VIP_VSYNC_0_EN_VIP_VSYNC                         8:8

#define LWE24D_VI_CONT_SYNCPT_VI2EPP_0                                         (0x1b)
// return INDX (set HOST_CLRD packet TYPE field to SYNCPT)
#define LWE24D_VI_CONT_SYNCPT_VI2EPP_0_INDX_VI2EPP                             7:0

// on host read bus every time VI2EPP condition is true and EN_VSYNC is set
#define LWE24D_VI_CONT_SYNCPT_VI2EPP_0_EN_VI2EPP                               8:8

#define LWE24D_VI_CONT_SYNCPT_CSI_PPA_FRAME_START_0                    (0x1c)
// return INDX (set HOST_CLRD packet TYPE field to SYNCPT)
#define LWE24D_VI_CONT_SYNCPT_CSI_PPA_FRAME_START_0_INDX_CSI_PPA_FRAME_START                     7:0

// on host read bus every time CSI_PPA_FRAME_START condition is true and EN_VSYNC is set
#define LWE24D_VI_CONT_SYNCPT_CSI_PPA_FRAME_START_0_EN_CSI_PPA_FRAME_START                       8:8

#define LWE24D_VI_CONT_SYNCPT_CSI_PPA_FRAME_END_0                      (0x1d)

// return INDX (set HOST_CLRD packet TYPE field to SYNCPT)
#define LWE24D_VI_CONT_SYNCPT_CSI_PPA_FRAME_END_0_INDX_CSI_PPA_FRAME_END                 7:0

// on host read bus every time CSI_PPA_FRAME_END condition is true and EN_VSYNC is set
#define LWE24D_VI_CONT_SYNCPT_CSI_PPA_FRAME_END_0_EN_CSI_PPA_FRAME_END                   8:8

#define LWE24D_VI_CONT_SYNCPT_CSI_PPB_FRAME_START_0                    (0x1e)

// return INDX (set HOST_CLRD packet TYPE field to SYNCPT)
#define LWE24D_VI_CONT_SYNCPT_CSI_PPB_FRAME_START_0_INDX_CSI_PPB_FRAME_START                     7:0

// on host read bus every time CSI_PPB_FRAME_START condition is true and EN_VSYNC is set
#define LWE24D_VI_CONT_SYNCPT_CSI_PPB_FRAME_START_0_EN_CSI_PPB_FRAME_START                       8:8


#define LWE24D_VI_CONT_SYNCPT_CSI_PPB_FRAME_END_0                      (0x1f)
// return INDX (set HOST_CLRD packet TYPE field to SYNCPT)
#define LWE24D_VI_CONT_SYNCPT_CSI_PPB_FRAME_END_0_INDX_CSI_PPB_FRAME_END                 7:0

// on host read bus every time CSI_PPB_FRAME_END condition is true and EN_VSYNC is set
#define LWE24D_VI_CONT_SYNCPT_CSI_PPB_FRAME_END_0_EN_CSI_PPB_FRAME_END                   8:8

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

#define LWE24D_VI_CTXSW_0                      (0x20)
// Current working class
#define LWE24D_VI_CTXSW_0_LWRR_CLASS                     9:0

// Automatically acknowledge any incoming context switch requests
#define LWE24D_VI_CTXSW_0_AUTO_ACK                       11:11
#define LWE24D_VI_CTXSW_0_AUTO_ACK_MANUAL                      (0)
#define LWE24D_VI_CTXSW_0_AUTO_ACK_AUTOACK                     (1)

// Current working channel, reset to 'invalid'
#define LWE24D_VI_CTXSW_0_LWRR_CHANNEL                   15:12

// Next requested class
#define LWE24D_VI_CTXSW_0_NEXT_CLASS                     25:16

// Next requested channel
#define LWE24D_VI_CTXSW_0_NEXT_CHANNEL                   31:28


#define LWE24D_VI_INTSTATUS_0                  (0x21)
// Context switch interrupt status (clear on write)
#define LWE24D_VI_INTSTATUS_0_CTXSW_INT                  0:0

// For Parallel VIP input, limitation for vsync and hsync has to be followed to avoid ISP hang for AP15:
// SW must always program parallel cameras (including the VIP pattern generator) in a way that
// avoids simultaneous hsync and vsync active edges. copied from bug:361730

#define LWE24D_VI_VI_INPUT_CONTROL_0                   (0x22)
// Host Input Enable   0= DISABLED
//   1= ENABLED
#define LWE24D_VI_VI_INPUT_CONTROL_0_HOST_INPUT_ENABLE                   0:0
#define LWE24D_VI_VI_INPUT_CONTROL_0_HOST_INPUT_ENABLE_DISABLED                        (0)
#define LWE24D_VI_VI_INPUT_CONTROL_0_HOST_INPUT_ENABLE_ENABLED                 (1)

// VIP Input Enable   0= DISABLED
//   1= ENABLED
// This bit turn on clocks for VIP input logic. This
//   bit has to be enabled before CAMERA_CONTROL's
//   VIP_ENABLE bit for any VIP logic to start!
#define LWE24D_VI_VI_INPUT_CONTROL_0_VIP_INPUT_ENABLE                    1:1
#define LWE24D_VI_VI_INPUT_CONTROL_0_VIP_INPUT_ENABLE_DISABLED                 (0)
#define LWE24D_VI_VI_INPUT_CONTROL_0_VIP_INPUT_ENABLE_ENABLED                  (1)

// Input port data Format  (effective if input source is VI Port)
//   0000= YUV422 or ITU-R BT.656
//   0001= Reserved 1
//   0010= Bayer Pattern, enables ISP
//   0011= Reserved 2
//   0100= Pattern A, written directly to memory
//   0101= Pattern B, written directly to memory
//   0110= Pattern C, written directly to memory
//   0111= Pattern C, do not remove the 0xFF, 0x000000002
//   1000= Pattern D, ISDB-T input
//   1001= YUV420NP, written directly to memory as YUV420P
//   1010= RGB565, written directly to EPP
//   1011= RGB888, written directly to EPP
//   1100= RGB444, written directly to EPP
//   1101= CSI,    written directly to CSI
//         For YUV420NP no cropping will be done.
//         For RGB565,RGB888,RGB444 written to EPP
//         all cropping will be done in the EPP.
#define LWE24D_VI_VI_INPUT_CONTROL_0_INPUT_PORT_FORMAT                   5:2
#define LWE24D_VI_VI_INPUT_CONTROL_0_INPUT_PORT_FORMAT_YUV422                  (0)
#define LWE24D_VI_VI_INPUT_CONTROL_0_INPUT_PORT_FORMAT_RESERVED_1                      (1)
#define LWE24D_VI_VI_INPUT_CONTROL_0_INPUT_PORT_FORMAT_BAYER                   (2)
#define LWE24D_VI_VI_INPUT_CONTROL_0_INPUT_PORT_FORMAT_RESERVED_2                      (3)
#define LWE24D_VI_VI_INPUT_CONTROL_0_INPUT_PORT_FORMAT_PATTERN_A                       (4)
#define LWE24D_VI_VI_INPUT_CONTROL_0_INPUT_PORT_FORMAT_PATTERN_B                       (5)
#define LWE24D_VI_VI_INPUT_CONTROL_0_INPUT_PORT_FORMAT_PATTERN_C                       (6)
#define LWE24D_VI_VI_INPUT_CONTROL_0_INPUT_PORT_FORMAT_PATTERN_C_RAW                   (7)
#define LWE24D_VI_VI_INPUT_CONTROL_0_INPUT_PORT_FORMAT_PATTERN_D                       (8)
#define LWE24D_VI_VI_INPUT_CONTROL_0_INPUT_PORT_FORMAT_YUV420                  (9)
#define LWE24D_VI_VI_INPUT_CONTROL_0_INPUT_PORT_FORMAT_RGB565                  (10)
#define LWE24D_VI_VI_INPUT_CONTROL_0_INPUT_PORT_FORMAT_RGB888                  (11)
#define LWE24D_VI_VI_INPUT_CONTROL_0_INPUT_PORT_FORMAT_RGB444                  (12)
#define LWE24D_VI_VI_INPUT_CONTROL_0_INPUT_PORT_FORMAT_CSI                     (13)

// Host data Format  (effective if input source is host)
//   00= Non-planar YUV422
//      (only Y-FIFO is used)
//   01= Planar YUV420
//      (Y-FIFO, U-FIFO, V-FIFO are used)
//   10= Bayer 8-bit  - enables ISP
//   11= Bayer 12-bit - enables ISP
#define LWE24D_VI_VI_INPUT_CONTROL_0_HOST_FORMAT                 7:6
#define LWE24D_VI_VI_INPUT_CONTROL_0_HOST_FORMAT_NONPLANAR                     (0)
#define LWE24D_VI_VI_INPUT_CONTROL_0_HOST_FORMAT_PLANAR                        (1)
#define LWE24D_VI_VI_INPUT_CONTROL_0_HOST_FORMAT_BAYER8                        (2)
#define LWE24D_VI_VI_INPUT_CONTROL_0_HOST_FORMAT_BAYER12                       (3)

// YUV Input Format This is applicable when input source is
// VI Port and format is YUV422/ITU-R BT.656
// or when input source is host and host
// format is non-planar YUV422.
//  8 bits per component
//   00= UYVY => Y1_V0_Y0_U0 MSB to LSB 32bit mapping
//   01= VYUY => Y1_U0_Y0_V0
//   10= YUYV => V0_Y1_U0_Y0
//   11= YVYU => U0_Y1_V0_Y0
#define LWE24D_VI_VI_INPUT_CONTROL_0_YUV_INPUT_FORMAT                    9:8
#define LWE24D_VI_VI_INPUT_CONTROL_0_YUV_INPUT_FORMAT_UYVY                     (0)
#define LWE24D_VI_VI_INPUT_CONTROL_0_YUV_INPUT_FORMAT_VYUY                     (1)
#define LWE24D_VI_VI_INPUT_CONTROL_0_YUV_INPUT_FORMAT_YUYV                     (2)
#define LWE24D_VI_VI_INPUT_CONTROL_0_YUV_INPUT_FORMAT_YVYU                     (3)

// Select a data source input to HOST (extension field).  (use when input source is host)
//  000= Source is selected with HOST_FORMAT field (backward compatible)
//  001= Bayer 10 bpp: 2 16-bit values packed into 32-bit, LSbit aligned {6'b0, bayer, 6'b0, bayer} (to ISP)
//  010= Bayer 14 bpp: 2 16-bit values packed into 32-bit, LSbit aligned {2'b0, bayer, 2'b0, bayer} (to ISP)
//  011= RGB565             (to EPP)
//  100= MSB Alpha + RGB888 (to EPP)
//  101= MSB Alpha + BGR888 (to EPP)
//  110= CSI                (to CSI)
//  111= reserved
// 22:13 reserved
#define LWE24D_VI_VI_INPUT_CONTROL_0_HOST_FORMAT_EXT                     12:10
#define LWE24D_VI_VI_INPUT_CONTROL_0_HOST_FORMAT_EXT_USE_HOST_FORMAT                   (0)
#define LWE24D_VI_VI_INPUT_CONTROL_0_HOST_FORMAT_EXT_BAYER10                   (1)
#define LWE24D_VI_VI_INPUT_CONTROL_0_HOST_FORMAT_EXT_BAYER14                   (2)
#define LWE24D_VI_VI_INPUT_CONTROL_0_HOST_FORMAT_EXT_RGB565                    (3)
#define LWE24D_VI_VI_INPUT_CONTROL_0_HOST_FORMAT_EXT_ARGB8888                  (4)
#define LWE24D_VI_VI_INPUT_CONTROL_0_HOST_FORMAT_EXT_ABGR8888                  (5)
#define LWE24D_VI_VI_INPUT_CONTROL_0_HOST_FORMAT_EXT_CSI                       (6)

// VHS input signal active edge which is used  as horizontal reference of input data.
//  VHS input ilwersion is evaluated first
//  before determining active edge.
//   0= Rising edge of VHS is active edge.
//      For ITU-R BT.656 data, leading edge of
//      horizontal sync is the active edge.
//   1= Falling edge of VHS is active edge
//      For ITU-R BT.656 data, trailing edge
//      of horizontal sync is the active edge.
#define LWE24D_VI_VI_INPUT_CONTROL_0_VHS_IN_EDGE                 23:23
#define LWE24D_VI_VI_INPUT_CONTROL_0_VHS_IN_EDGE_RISING                        (0)
#define LWE24D_VI_VI_INPUT_CONTROL_0_VHS_IN_EDGE_FALLING                       (1)

// VVS input signal active edge which is used  as vertical reference of input data
//  VVS input ilwersion is evaluated first
//  before determining active edge.
//   0= Rising edge of VVS is active edge
//      For ITU-R BT.656 data, leading edge of
//      vertical sync is the active edge.
//   1= Falling edge of VVS is active edge
//      For ITU-R BT.656 data, trailing edge
//      of vertical sync is the active edge.
#define LWE24D_VI_VI_INPUT_CONTROL_0_VVS_IN_EDGE                 24:24
#define LWE24D_VI_VI_INPUT_CONTROL_0_VVS_IN_EDGE_RISING                        (0)
#define LWE24D_VI_VI_INPUT_CONTROL_0_VVS_IN_EDGE_FALLING                       (1)

// Horizontal and Vertical Sync Format  (effective if VIDEO_SOURCE is VIP)
//   00= horizontal sync comes from VHS pin
//       and vertical sync comes from VVS pin
//       consistent with standard YUV422 data
//       format.
//       In this case, VHS_Input_Control and
//       VVS_Input_Control must be enabled.
//   01= horizontal and vertical syncs are
//       decoded from the received video data
//       bytes as specified in ITU-R BT.656
//       (CCIR656) standard.
//   10= horizontal and vertical syncs are
//       generated internally and they are
//       output on VHS and VVS pins if VHS and
//       VVS are in output mode.
#define LWE24D_VI_VI_INPUT_CONTROL_0_SYNC_FORMAT                 26:25
#define LWE24D_VI_VI_INPUT_CONTROL_0_SYNC_FORMAT_YUV422                        (0)
#define LWE24D_VI_VI_INPUT_CONTROL_0_SYNC_FORMAT_ITU656                        (1)
#define LWE24D_VI_VI_INPUT_CONTROL_0_SYNC_FORMAT_INTHVS                        (2)

// Interlaced video Field Detection  (effective if VIDEO_SOURCE is VIP)
//   0= Disabled (top field only)
//   1= Enabled
//      When H/V syncs are decoded per ITU-R
//      BT.656 standard, odd/even field is
//      detected from the control bytes.
//      When H/V syncs come from VHS/VVS pins
//      (YUV422), odd/even field is detected
//      from the position of VVS active edge
//      with respect to VHS active pulse.
//      This bit should be disabled for non-
//      interlaced source or when H/V syncs
//      are generated internally.
//  If VIDEO_SOURCE is HOST, field information
//  is always specified by host.
#define LWE24D_VI_VI_INPUT_CONTROL_0_FIELD_DETECT                        27:27
#define LWE24D_VI_VI_INPUT_CONTROL_0_FIELD_DETECT_DISABLED                     (0)
#define LWE24D_VI_VI_INPUT_CONTROL_0_FIELD_DETECT_ENABLED                      (1)

// Odd/Even Field type  (effective for interlaced video source)
//   0= Top field is odd field
//   1= Top field is even field
#define LWE24D_VI_VI_INPUT_CONTROL_0_FIELD_TYPE                  28:28
#define LWE24D_VI_VI_INPUT_CONTROL_0_FIELD_TYPE_TOPODD                 (0)
#define LWE24D_VI_VI_INPUT_CONTROL_0_FIELD_TYPE_TOPEVEN                        (1)

// Horizontal Counter   0= Enabled
//   1= Disabled (reset to 0)
#define LWE24D_VI_VI_INPUT_CONTROL_0_H_COUNTER                   29:29
#define LWE24D_VI_VI_INPUT_CONTROL_0_H_COUNTER_ENABLED                 (0)
#define LWE24D_VI_VI_INPUT_CONTROL_0_H_COUNTER_DISABLED                        (1)

// Vertical Counter   0= Enabled
//   1= Disabled (reset to 0)
#define LWE24D_VI_VI_INPUT_CONTROL_0_V_COUNTER                   30:30
#define LWE24D_VI_VI_INPUT_CONTROL_0_V_COUNTER_ENABLED                 (0)
#define LWE24D_VI_VI_INPUT_CONTROL_0_V_COUNTER_DISABLED                        (1)


#define LWE24D_VI_VI_CORE_CONTROL_0                    (0x23)
// Output to ISP  Enable data output to ISP
//   00= Output to ISP is disabled
//   01= Parallel Video Input Port data
//   10= Host I/F data
//   11= reserved
#define LWE24D_VI_VI_CORE_CONTROL_0_OUTPUT_TO_ISP                        1:0
#define LWE24D_VI_VI_CORE_CONTROL_0_OUTPUT_TO_ISP_DISABLED                     (0)
#define LWE24D_VI_VI_CORE_CONTROL_0_OUTPUT_TO_ISP_VIP                  (1)
#define LWE24D_VI_VI_CORE_CONTROL_0_OUTPUT_TO_ISP_HOST                 (2)

// Output to EPP enable  VI can output a YUV pixel stream to
//  Encoder Pre-Processor (EPP) module
//   000= Output to EPP is disabled
//   001= YUV444 stream after down-scaling
//   010= YUV444 stream before down-scaling
//        WARNING: FOR YUV444PRE, only the selects
//        in INPUT_TO_CORE are supported.  Selects from
//        INPUT_TO_CORE_EXT are not supported since they
//        are duplicated in the CSI* selections of this field.
//   011= YUV444 stream from ISP, no LPF or down-scaling
//   100= RGB565,RGB444,RGB888 from VIP, no LPF or down-scaling
//   101= RGB565,RGB888 from Host
//   110= CSI_PPA
//   111= CSI_PPB
#define LWE24D_VI_VI_CORE_CONTROL_0_OUTPUT_TO_EPP                        4:2
#define LWE24D_VI_VI_CORE_CONTROL_0_OUTPUT_TO_EPP_DISABLED                     (0)
#define LWE24D_VI_VI_CORE_CONTROL_0_OUTPUT_TO_EPP_YUV444POST                   (1)
#define LWE24D_VI_VI_CORE_CONTROL_0_OUTPUT_TO_EPP_YUV444PRE                    (2)
#define LWE24D_VI_VI_CORE_CONTROL_0_OUTPUT_TO_EPP_YUV444ISP                    (3)
#define LWE24D_VI_VI_CORE_CONTROL_0_OUTPUT_TO_EPP_RGB                  (4)
#define LWE24D_VI_VI_CORE_CONTROL_0_OUTPUT_TO_EPP_HOST_RGB                     (5)
#define LWE24D_VI_VI_CORE_CONTROL_0_OUTPUT_TO_EPP_CSI_PPA                      (6)
#define LWE24D_VI_VI_CORE_CONTROL_0_OUTPUT_TO_EPP_CSI_PPB                      (7)

// Downsample from YUV444 to YUV422   00 = Cosited, take even UV's for each two Y's.
//   01 = Cosited, take odd UV's for each two Y's. (Not implemented)
//   10 = Non Cosited, take even U and odd V, use for Bayer passthru
//   11 = Averaged, average the odd and even UVs. (Not Implemented)
#define LWE24D_VI_VI_CORE_CONTROL_0_ISP_DOWNSAMPLE                       6:5
#define LWE24D_VI_VI_CORE_CONTROL_0_ISP_DOWNSAMPLE_COSITED_EVEN                        (0)
#define LWE24D_VI_VI_CORE_CONTROL_0_ISP_DOWNSAMPLE_COSITED_ODD                 (1)
#define LWE24D_VI_VI_CORE_CONTROL_0_ISP_DOWNSAMPLE_NONCOSITED                  (2)
#define LWE24D_VI_VI_CORE_CONTROL_0_ISP_DOWNSAMPLE_AVERAGED                    (3)

// Input to VI Core  Select between possible data input sources
//   00= Parallel Video Input Port data
//   01= Host I/F data
//   10= ISP data, from 444 to 422 colwerter
//   11= reserved
#define LWE24D_VI_VI_CORE_CONTROL_0_INPUT_TO_CORE                        9:8
#define LWE24D_VI_VI_CORE_CONTROL_0_INPUT_TO_CORE_VIP                  (0)
#define LWE24D_VI_VI_CORE_CONTROL_0_INPUT_TO_CORE_HOST                 (1)
#define LWE24D_VI_VI_CORE_CONTROL_0_INPUT_TO_CORE_ISP                  (2)

// Planar Colwersion Module Input select   0= YUV422 after down-scaling, POST core
//   1= YUV422 before down-scaling, PRE core
//
#define LWE24D_VI_VI_CORE_CONTROL_0_PLANAR_COLW_INPUT_SEL                        10:10
#define LWE24D_VI_VI_CORE_CONTROL_0_PLANAR_COLW_INPUT_SEL_YUV422POST                   (0)
#define LWE24D_VI_VI_CORE_CONTROL_0_PLANAR_COLW_INPUT_SEL_YUV422PRE                    (1)

// Color Space Colwersion Input select   0= YUV422 after down-scaling, POST core
//   1= YUV422 before down-scaling, PRE core
// 15:12 reserved
#define LWE24D_VI_VI_CORE_CONTROL_0_CSC_INPUT_SEL                        11:11
#define LWE24D_VI_VI_CORE_CONTROL_0_CSC_INPUT_SEL_YUV422POST                   (0)
#define LWE24D_VI_VI_CORE_CONTROL_0_CSC_INPUT_SEL_YUV422PRE                    (1)

// Horizontal Averaging   0= disabled, H_DOWNSCALING can be used
//      to enable horizontal downscaling
//   1= enabled, H_DOWNSCALING is ignored
//      and horizontal downscaling is
//      controlled by H_AVG_FACTOR
#define LWE24D_VI_VI_CORE_CONTROL_0_H_AVERAGING                  16:16
#define LWE24D_VI_VI_CORE_CONTROL_0_H_AVERAGING_DISABLED                       (0)
#define LWE24D_VI_VI_CORE_CONTROL_0_H_AVERAGING_ENABLED                        (1)

// Horizontal Down-scaling  (effective if H_AVERAGING is DISABLED)
//   0= disabled
//   1= enabled and controlled by H_DOWN_M
//      and H_DOWN_N parameters
#define LWE24D_VI_VI_CORE_CONTROL_0_H_DOWNSCALING                        17:17
#define LWE24D_VI_VI_CORE_CONTROL_0_H_DOWNSCALING_DISABLED                     (0)
#define LWE24D_VI_VI_CORE_CONTROL_0_H_DOWNSCALING_ENABLED                      (1)

// Vertical Averaging   0= disabled, V_DOWNSCALING can be used
//      to enable vertical downscaling
//   1= enabled, V_DOWNSCALING is ignored
//      and vertical downscaling is
//      controlled by V_AVG_FACTOR
#define LWE24D_VI_VI_CORE_CONTROL_0_V_AVERAGING                  18:18
#define LWE24D_VI_VI_CORE_CONTROL_0_V_AVERAGING_DISABLED                       (0)
#define LWE24D_VI_VI_CORE_CONTROL_0_V_AVERAGING_ENABLED                        (1)

// Vertical Down-scaling  (effective if V_AVERAGING is DISABLED)
//   0= disabled
//   1= enabled and controlled by V_DOWN_M
//      and V_DOWN_N parameters
#define LWE24D_VI_VI_CORE_CONTROL_0_V_DOWNSCALING                        19:19
#define LWE24D_VI_VI_CORE_CONTROL_0_V_DOWNSCALING_DISABLED                     (0)
#define LWE24D_VI_VI_CORE_CONTROL_0_V_DOWNSCALING_ENABLED                      (1)

// ISP Host data stall capability is enabled by default  Use this bit to disable the host data stall capability
//   0= disabled - default allows for VI to turn off
//                 the ISP clock to stall the Host.
//   1= enabled - to turn off the VI's ability to stall the Host
//                when data from ISP comes from Host.
//
#define LWE24D_VI_VI_CORE_CONTROL_0_ISP_HOST_STALL_OFF                   20:20
#define LWE24D_VI_VI_CORE_CONTROL_0_ISP_HOST_STALL_OFF_DISABLED                        (0)
#define LWE24D_VI_VI_CORE_CONTROL_0_ISP_HOST_STALL_OFF_ENABLED                 (1)

// Select a data source output to ISP (extension field).
//   000= Source is selected with OUTPUT_TO_ISP field (backward compatible)
//   001= CSI Pixel Parser A
//   010= CSI Pixel Parser B
#define LWE24D_VI_VI_CORE_CONTROL_0_OUTPUT_TO_ISP_EXT                    23:21
#define LWE24D_VI_VI_CORE_CONTROL_0_OUTPUT_TO_ISP_EXT_USE_OUTPUT_TO_ISP                        (0)
#define LWE24D_VI_VI_CORE_CONTROL_0_OUTPUT_TO_ISP_EXT_CSI_PPA                  (1)
#define LWE24D_VI_VI_CORE_CONTROL_0_OUTPUT_TO_ISP_EXT_CSI_PPB                  (2)

// Select a data source input to core (extension field).
//   000= Source is selected with INPUT_TO_CORE field (backward compatible)
//   001= CSI_PPA data in YUV444NP format
//   010= CSI_PPA data in YUV422NP format
//   011= CSI_PPB data in YUV444NP format
//   100= CSI_PPB data in YUV422NP format
#define LWE24D_VI_VI_CORE_CONTROL_0_INPUT_TO_CORE_EXT                    26:24
#define LWE24D_VI_VI_CORE_CONTROL_0_INPUT_TO_CORE_EXT_USE_INPUT_TO_CORE                        (0)
#define LWE24D_VI_VI_CORE_CONTROL_0_INPUT_TO_CORE_EXT_CSI_PPA_YUV444                   (1)
#define LWE24D_VI_VI_CORE_CONTROL_0_INPUT_TO_CORE_EXT_CSI_PPA_YUV422                   (2)
#define LWE24D_VI_VI_CORE_CONTROL_0_INPUT_TO_CORE_EXT_CSI_PPB_YUV444                   (3)
#define LWE24D_VI_VI_CORE_CONTROL_0_INPUT_TO_CORE_EXT_CSI_PPB_YUV422                   (4)


#define LWE24D_VI_VI_FIRST_OUTPUT_CONTROL_0                    (0x24)
// Output data Format  Take from the CSC Unit:
//   000= 16-bit RGB (B5G6R5)
//   001= 16-bit RGB (B5G6R5) Dithered
//        (This is lwrrently NOT implemented)
//   010= 24-bit RGB (B8G8R8)
//  Take from the YUV422 Core output path:
//      (Same thing as using YUV422PRE and YUV_SOURCE==CORE_OUTPUT)
//   011= YUV422 non-planar (U8Y8V8Y8) after down-scaling, POST
//  Take from the YUV422 paths: (see YUV_SOURCE field)
//   100= YUV422 non-planar (U8Y8V8Y8) before down-scaling, PRE
//   101= YUV422 Planar
//   110= YUV420 Planar
//   111= YUV420 Planar with Averaging
//        (UV is averaged for each line pair)
// 7:3 reserved
#define LWE24D_VI_VI_FIRST_OUTPUT_CONTROL_0_OUTPUT_FORMAT                        2:0
#define LWE24D_VI_VI_FIRST_OUTPUT_CONTROL_0_OUTPUT_FORMAT_RGB16                        (0)
#define LWE24D_VI_VI_FIRST_OUTPUT_CONTROL_0_OUTPUT_FORMAT_RGB16D                       (1)
#define LWE24D_VI_VI_FIRST_OUTPUT_CONTROL_0_OUTPUT_FORMAT_RGB24                        (2)
#define LWE24D_VI_VI_FIRST_OUTPUT_CONTROL_0_OUTPUT_FORMAT_YUV422POST                   (3)
#define LWE24D_VI_VI_FIRST_OUTPUT_CONTROL_0_OUTPUT_FORMAT_YUV422PRE                    (4)
#define LWE24D_VI_VI_FIRST_OUTPUT_CONTROL_0_OUTPUT_FORMAT_YUV422P                      (5)
#define LWE24D_VI_VI_FIRST_OUTPUT_CONTROL_0_OUTPUT_FORMAT_YUV420P                      (6)
#define LWE24D_VI_VI_FIRST_OUTPUT_CONTROL_0_OUTPUT_FORMAT_YUV420PA                     (7)

// For Planar Output Only, enabling this register  duplicates the last pixel of each line when
//  the output width is set to an odd number of pixels.
//  Used when JPEGE/MPEGE which requires valid data filled
//  to the word(16-bit) boundary.
//  The Buffer Horizontal Size (Line Stride) must be
//  set to accomodate the extra pixel.
//  Example: Disabled - y0,y1,y2,y3,y4
//           Enabled - y0,y1,y2,y3,y4,y4
// 15:9 reserved
#define LWE24D_VI_VI_FIRST_OUTPUT_CONTROL_0_LAST_PIXEL_DUPLICATION                       8:8
#define LWE24D_VI_VI_FIRST_OUTPUT_CONTROL_0_LAST_PIXEL_DUPLICATION_DISABLED                    (0)
#define LWE24D_VI_VI_FIRST_OUTPUT_CONTROL_0_LAST_PIXEL_DUPLICATION_ENABLED                     (1)

// Output Byte Swap  (effective if input source is host)
//   0= disabled
//   1= enabled
#define LWE24D_VI_VI_FIRST_OUTPUT_CONTROL_0_OUTPUT_BYTE_SWAP                     16:16
#define LWE24D_VI_VI_FIRST_OUTPUT_CONTROL_0_OUTPUT_BYTE_SWAP_DISABLED                  (0)
#define LWE24D_VI_VI_FIRST_OUTPUT_CONTROL_0_OUTPUT_BYTE_SWAP_ENABLED                   (1)

// YUV Output Format This is applicable when output format is
// non-planar YUV422.
//   00= UYVY => Y1_V0_Y1_U0 MSB to LSB 32bit mapping
//   01= VYUY => Y1_U0_Y1_V0
//   10= YUYV => V0_Y1_U0_Y0
//   11= YVYU => U0_Y1_V0_Y0
#define LWE24D_VI_VI_FIRST_OUTPUT_CONTROL_0_YUV_OUTPUT_FORMAT                    18:17
#define LWE24D_VI_VI_FIRST_OUTPUT_CONTROL_0_YUV_OUTPUT_FORMAT_UYVY                     (0)
#define LWE24D_VI_VI_FIRST_OUTPUT_CONTROL_0_YUV_OUTPUT_FORMAT_VYUY                     (1)
#define LWE24D_VI_VI_FIRST_OUTPUT_CONTROL_0_YUV_OUTPUT_FORMAT_YUYV                     (2)
#define LWE24D_VI_VI_FIRST_OUTPUT_CONTROL_0_YUV_OUTPUT_FORMAT_YVYU                     (3)

//  H-direction in internal memory
#define LWE24D_VI_VI_FIRST_OUTPUT_CONTROL_0_H_DIRECTION                  19:19

//  V-direction in internal memory
#define LWE24D_VI_VI_FIRST_OUTPUT_CONTROL_0_V_DIRECTION                  20:20

//  XY_SWAP IS NO LONGER SUPPORTED
#define LWE24D_VI_VI_FIRST_OUTPUT_CONTROL_0_XY_SWAP                      21:21


#define LWE24D_VI_VI_SECOND_OUTPUT_CONTROL_0                   (0x25)
// Secondary Output to MC  Use case: when VI needs to send decimated preview data
//  and at the same time send non-decimated data
//  to the memory for StretchBLT, meanwhile the StretchBLT
//  is sending EPP stretched data to be encoded.
//  Only YUV422, RGB888, RGB565 is supported
//
//  Take from the CSC Unit
//  0000= 16-bit RGB (B5G6R5), all RGB data can be pre or
//        post decimated depending on mux select programming
//        on the input to the Color Space Colwerter
//  0001= 16-bit RGB (B5G6R5) Dithered
//        (This is lwrrently NOT implemented)
//  0010= 24-bit RGB (B8G8R8)
//  Take from the YUV422 Core output path:
//      (Same thing as using YUV422PRE and YUV_SOURCE==CORE_OUTPUT)
//  0011= YUV422 stream after down-scaling, POST
//  Take from the YUV422 paths: (see YUV_SOURCE field)
//  0100= YUV422 stream before down-scaling, PRE
//  Take from the WriteBuffer interface logic, which is used for JPEG Stream
//  0101= JPEG Stream (Pattern A,B,C)
//  0110= VIP Bayer     direct to memory as a 16-bit value {6'b0, VIP_pad[9:0]}
//  0111= CSI_PPA Bayer direct to memory as a 16-bit value {6'b0, CSI_SVD[15:6]}
//  1000= CSI_PPB Bayer direct to memory as a 16-bit value {6'b0, CSI_SVD[15:6]}
//  VIP_BAYER_DIRECT: Bayer data is written unmodified to memory
//  as a 16-bit quantity.  Bit0 of incoming data is placed in
//  bit0 of the 16-bit memory location. Upper bits are padded with 0.
// 15:4 reserved
#define LWE24D_VI_VI_SECOND_OUTPUT_CONTROL_0_SECOND_OUTPUT_FORMAT                        3:0
#define LWE24D_VI_VI_SECOND_OUTPUT_CONTROL_0_SECOND_OUTPUT_FORMAT_RGB16                        (0)
#define LWE24D_VI_VI_SECOND_OUTPUT_CONTROL_0_SECOND_OUTPUT_FORMAT_RGB16D                       (1)
#define LWE24D_VI_VI_SECOND_OUTPUT_CONTROL_0_SECOND_OUTPUT_FORMAT_RGB24                        (2)
#define LWE24D_VI_VI_SECOND_OUTPUT_CONTROL_0_SECOND_OUTPUT_FORMAT_YUV422POST                   (3)
#define LWE24D_VI_VI_SECOND_OUTPUT_CONTROL_0_SECOND_OUTPUT_FORMAT_YUV422PRE                    (4)
#define LWE24D_VI_VI_SECOND_OUTPUT_CONTROL_0_SECOND_OUTPUT_FORMAT_JPEG_STREAM                  (5)
#define LWE24D_VI_VI_SECOND_OUTPUT_CONTROL_0_SECOND_OUTPUT_FORMAT_VIP_BAYER                    (6)
#define LWE24D_VI_VI_SECOND_OUTPUT_CONTROL_0_SECOND_OUTPUT_FORMAT_CSI_PPA_BAYER                        (7)
#define LWE24D_VI_VI_SECOND_OUTPUT_CONTROL_0_SECOND_OUTPUT_FORMAT_CSI_PPB_BAYER                        (8)
#define LWE24D_VI_VI_SECOND_OUTPUT_CONTROL_0_SECOND_OUTPUT_FORMAT_VIP_BAYER_DIRECT                     (9)

// Output Byte Swap  (effective if input source is host)
//   0= disabled
//   1= enabled
#define LWE24D_VI_VI_SECOND_OUTPUT_CONTROL_0_SECOND_OUTPUT_BYTE_SWAP                     16:16
#define LWE24D_VI_VI_SECOND_OUTPUT_CONTROL_0_SECOND_OUTPUT_BYTE_SWAP_DISABLED                  (0)
#define LWE24D_VI_VI_SECOND_OUTPUT_CONTROL_0_SECOND_OUTPUT_BYTE_SWAP_ENABLED                   (1)

// YUV Second Output Format This is applicable when output format is
// non-planar YUV422.
//   00= UYVY => Y1_V0_Y1_U0 MSB to LSB 32bit mapping
//   01= VYUY => Y1_U0_Y1_V0
//   10= YUYV => V0_Y1_U0_Y0
//   11= YVYU => U0_Y1_V0_Y0
#define LWE24D_VI_VI_SECOND_OUTPUT_CONTROL_0_YUV_SECOND_OUTPUT_FORMAT                    18:17
#define LWE24D_VI_VI_SECOND_OUTPUT_CONTROL_0_YUV_SECOND_OUTPUT_FORMAT_UYVY                     (0)
#define LWE24D_VI_VI_SECOND_OUTPUT_CONTROL_0_YUV_SECOND_OUTPUT_FORMAT_VYUY                     (1)
#define LWE24D_VI_VI_SECOND_OUTPUT_CONTROL_0_YUV_SECOND_OUTPUT_FORMAT_YUYV                     (2)
#define LWE24D_VI_VI_SECOND_OUTPUT_CONTROL_0_YUV_SECOND_OUTPUT_FORMAT_YVYU                     (3)

//  Second output's H-direction in internal memory
#define LWE24D_VI_VI_SECOND_OUTPUT_CONTROL_0_SECOND_H_DIRECTION                  19:19

//  Second output's V-direction in internal memory
#define LWE24D_VI_VI_SECOND_OUTPUT_CONTROL_0_SECOND_V_DIRECTION                  20:20

//  XY_SWAP IS NO LONGER SUPPORTED
#define LWE24D_VI_VI_SECOND_OUTPUT_CONTROL_0_SECOND_XY_SWAP                      21:21

// Input Frame Width and Height give the total input data dimensions.  The VI input stage will lwll/clip
// pixels outside the Active Region (see register VI_HOST_H_ACTIVE & VI_HOST_V_ACTIVE).  The amount of data
// per frame is expected to be INPUT_WIDTH * INPUT_HEIGHT * the bytes per pixel (determined from the
// INPUT_HOST_FORMAT). For Planar, the BPP is 1 for the Y fifo, 1/2 for U and V. For non planar it is 2.
// The Bayer data is treated as 1 byte per pixel, so if it is more, then the input width and the H_ACTIVE
// should be scaled accordingly, so that internally generated hsync and vsyncs for ISP are correct.
// For Bayer input, it is important to insert blanking data for horizontal and vertical, allowing ISP to do
// side band callwlations.

#define LWE24D_VI_HOST_INPUT_FRAME_SIZE_0                      (0x26)
// Specifies in terms of pixels the width of
// the input data coming from host.
#define LWE24D_VI_HOST_INPUT_FRAME_SIZE_0_INPUT_FRAME_WIDTH                      12:0

// Host Input Frame Height
// Specifies in terms of lines the height of
// the input data coming from host.
#define LWE24D_VI_HOST_INPUT_FRAME_SIZE_0_INPUT_FRAME_HEIGHT                     28:16

// This register defines the horizontal active area of the input video source with respect to
// the internally generated horizontal sync.  (This is for data coming in from host.)

#define LWE24D_VI_HOST_H_ACTIVE_0                      (0x27)
// Horizontal Active Start (offset to active)
//  This parameter specifies the number of
//  pixels to be discarded until the first
//  active pixel. If programmed to 0, the
//  first active pixel is the first pixel popped
//  from the Host YUV FIFO.
#define LWE24D_VI_HOST_H_ACTIVE_0_HOST_H_ACTIVE_START                    12:0

// Horizontal Active Period
//  This parameter specifies the number of
//  pixels in the horizontal active area.
//  H_ACTIVE_START + H_ACTIVE_PERIOD should be
//  less than 2^LW_VI_H_IN (or 8192) This parameter
//  should be programmed with an even number
//  (bit 16 is ignored internally).
#define LWE24D_VI_HOST_H_ACTIVE_0_HOST_H_ACTIVE_PERIOD                   28:16

// This register defines the vertical active area of the input video source with respect to
// the internally generated vertical sync.  (This is for data coming in from host.)

#define LWE24D_VI_HOST_V_ACTIVE_0                      (0x28)
// Vertical Active Start (offset to active)
//  This parameter specifies the number of
//  horizontal sync active edges from vertical
//  sync active edge to the first vertical
//  active line. If programmed to 0, the
//  first active line starts after the first
//  horizontal sync active edge following
//  the vertical sync active edge.
#define LWE24D_VI_HOST_V_ACTIVE_0_HOST_V_ACTIVE_START                    12:0

// Vertical Active Period
//  This parameter specifies the number of
//  lines in the vertical active area.
//  V_ACTIVE_START + V_ACTIVE_PERIOD should be
//  less than 2^LW_VI_V_IN (or 8192).
#define LWE24D_VI_HOST_V_ACTIVE_0_HOST_V_ACTIVE_PERIOD                   28:16

// This register defines the horizontal active area of the input video source with respect to
//  horizontal sync. (This is for VIP data.)

#define LWE24D_VI_VIP_H_ACTIVE_0                       (0x29)
// Horizontal Active Start (offset to active)
//  This parameter specifies the number of
//  clock active edges from horizontal
//  sync active edge to the first horizontal
//  active pixel. If programmed to 0, the
//  first active line starts after the first
//  active clock edge following the horizontal
//  sync active edge.
#define LWE24D_VI_VIP_H_ACTIVE_0_VIP_H_ACTIVE_START                      12:0

// Horizontal Active Period
//  This parameter specifies the number of
//  pixels in the horizontal active area.
//  Bug #178631
//  The value is the END of the active region,
//  so PERIOD-START = active area
//  This parameter should be programmed
//  with an even number
#define LWE24D_VI_VIP_H_ACTIVE_0_VIP_H_ACTIVE_PERIOD                     28:16

// This register defines the vertical active area of the input video source with respect to
//  vertical sync. (This is for VIP data.)

#define LWE24D_VI_VIP_V_ACTIVE_0                       (0x2a)
// Vertical Active Start (offset to active)
//  This parameter specifies the number of
//  horizontal sync active edges from vertical
//  sync active edge to the first vertical
//  active line. If programmed to 0, the
//  first active line starts after the first
//  horizontal sync active edge following
//  the vertical sync active edge.
#define LWE24D_VI_VIP_V_ACTIVE_0_VIP_V_ACTIVE_START                      12:0

// Vertical Active Period
//  This parameter specifies the number of
//  lines in the vertical active area.
//  Bug #178631
//  The value is the END of the active region,
//  so PERIOD-START = active area
#define LWE24D_VI_VIP_V_ACTIVE_0_VIP_V_ACTIVE_PERIOD                     28:16


// For all fields:
//   00= Disabled
//   01= First memory
//   10= Second memory
//   11= not defined
#define LWE24D_VI_VI_PEER_CONTROL_0                    (0x2b)
// VI to Display Control Bus enable  VI will send a valid buffer signal
//  along with Y,U,V buffer addresses
//  and Frame Start and Frame End
#define LWE24D_VI_VI_PEER_CONTROL_0_DISPLAY_CONTROL                      1:0
#define LWE24D_VI_VI_PEER_CONTROL_0_DISPLAY_CONTROL_DISABLED                   (0)
#define LWE24D_VI_VI_PEER_CONTROL_0_DISPLAY_CONTROL_FIRST                      (1)
#define LWE24D_VI_VI_PEER_CONTROL_0_DISPLAY_CONTROL_SECOND                     (2)

// VI to JPEGE & MPEGE Control Bus enable  VI will send a valid buffer signal
//  along with buffer index
//  and Frame Start and Frame End
#define LWE24D_VI_VI_PEER_CONTROL_0_ENCODER_CONTROL                      3:2
#define LWE24D_VI_VI_PEER_CONTROL_0_ENCODER_CONTROL_DISABLED                   (0)
#define LWE24D_VI_VI_PEER_CONTROL_0_ENCODER_CONTROL_FIRST                      (1)
#define LWE24D_VI_VI_PEER_CONTROL_0_ENCODER_CONTROL_SECOND                     (2)

// VI to StretchBLT Control Bus enable  VI will send a valid buffer signal
//  along with buffer index
//  and Frame Start and Frame End
//  The VI to SB control bus is separate from
//  the VI to JPEGE/MPEGE bus.  This control
//  bus is controlled by the "2nd Output to
//  MC" write client interface.
#define LWE24D_VI_VI_PEER_CONTROL_0_SB_CONTROL                   5:4
#define LWE24D_VI_VI_PEER_CONTROL_0_SB_CONTROL_DISABLED                        (0)
#define LWE24D_VI_VI_PEER_CONTROL_0_SB_CONTROL_FIRST                   (1)
#define LWE24D_VI_VI_PEER_CONTROL_0_SB_CONTROL_SECOND                  (2)

// VI to Display B Control Bus enable  VI will send a valid buffer signal
//  along with Y,U,V buffer addresses
//  and Frame Start and Frame End
#define LWE24D_VI_VI_PEER_CONTROL_0_DISPLAY_B_CONTROL                    7:6
#define LWE24D_VI_VI_PEER_CONTROL_0_DISPLAY_B_CONTROL_DISABLED                 (0)
#define LWE24D_VI_VI_PEER_CONTROL_0_DISPLAY_B_CONTROL_FIRST                    (1)
#define LWE24D_VI_VI_PEER_CONTROL_0_DISPLAY_B_CONTROL_SECOND                   (2)


#define LWE24D_VI_VI_DMA_SELECT_0                      (0x2c)
// Host DMA Request enable at end of block  Request to host DMA can be enabled every
//  time a block of video input data is
//  written to memory.
//   00= Disabled
//   01= Write Buffer DMA for RAW data stream
//   10= First memory
//   11= Second memory
#define LWE24D_VI_VI_DMA_SELECT_0_DMA_REQUEST                    1:0
#define LWE24D_VI_VI_DMA_SELECT_0_DMA_REQUEST_DISABLED                 (0)
#define LWE24D_VI_VI_DMA_SELECT_0_DMA_REQUEST_STREAM                   (1)
#define LWE24D_VI_VI_DMA_SELECT_0_DMA_REQUEST_FIRST                    (2)
#define LWE24D_VI_VI_DMA_SELECT_0_DMA_REQUEST_SECOND                   (3)


#define LWE24D_VI_HOST_DMA_WRITE_BUFFER_0                      (0x2d)
// Buffer Size
#define LWE24D_VI_HOST_DMA_WRITE_BUFFER_0_BUFFER_SIZE                    15:0

// Buffer Number
#define LWE24D_VI_HOST_DMA_WRITE_BUFFER_0_BUFFER_NUMBER                  24:16

// DMA Enable
#define LWE24D_VI_HOST_DMA_WRITE_BUFFER_0_DMA_ENABLE                     25:25
#define LWE24D_VI_HOST_DMA_WRITE_BUFFER_0_DMA_ENABLE_DISABLED                  (0)
#define LWE24D_VI_HOST_DMA_WRITE_BUFFER_0_DMA_ENABLE_ENABLED                   (1)

// Data source selection 00= VIP     (backward compatible)
// 01= CSI_PPA
// 10= CSI_PPB
#define LWE24D_VI_HOST_DMA_WRITE_BUFFER_0_SOURCE_SEL                     27:26
#define LWE24D_VI_HOST_DMA_WRITE_BUFFER_0_SOURCE_SEL_VIP                       (0)
#define LWE24D_VI_HOST_DMA_WRITE_BUFFER_0_SOURCE_SEL_CSI_PPA                   (1)
#define LWE24D_VI_HOST_DMA_WRITE_BUFFER_0_SOURCE_SEL_CSI_PPB                   (2)


#define LWE24D_VI_HOST_DMA_BASE_ADDRESS_0                      (0x2e)
// Base Address
#define LWE24D_VI_HOST_DMA_BASE_ADDRESS_0_DMA_BASE_ADDR                  31:0


#define LWE24D_VI_HOST_DMA_WRITE_BUFFER_STATUS_0                       (0x2f)
// Read Only
#define LWE24D_VI_HOST_DMA_WRITE_BUFFER_STATUS_0_WB_STATUS                       26:0


#define LWE24D_VI_HOST_DMA_WRITE_PEND_BUFCOUNT_0                       (0x30)
// Read Only
#define LWE24D_VI_HOST_DMA_WRITE_PEND_BUFCOUNT_0_PEND_BUFCOUNT                   8:0

// FIRST OUTPUT Registers
// These registers are used to setup the first of two memory outputs for VI
// Address Y, U, V; Frame size; Count; Size (line stride and block height); and Buffer Stride

#define LWE24D_VI_VB0_START_ADDRESS_FIRST_0                    (0x31)
//  This is byte address of video buffer 0 if
//  output data format is RGB or YUV non-planar.
//  This is byte address of video buffer 0
//  Y-plane if output data format is YUV planar.
#define LWE24D_VI_VB0_START_ADDRESS_FIRST_0_VB0_START_ADDRESS_1                  31:0

// BASE address is used in Tiling mode. BASE address always points to the left_upper cornor
// of a surface. A surface can contain multiple buffers, in cirlwlar_buffer case.
// Write to the BASE address register with cause corresponding internal buffer index set back
// to zero.

#define LWE24D_VI_VB0_BASE_ADDRESS_FIRST_0                     (0x32)
//  This is the first byte address of video
//  buffer 0.
//  This is byte address of video buffer 0
//  Y-plane if output data format is planar.
#define LWE24D_VI_VB0_BASE_ADDRESS_FIRST_0_VB0_BASE_ADDRESS_1                    31:0


#define LWE24D_VI_VB0_START_ADDRESS_U_0                        (0x33)
//  This is byte address of video buffer 0
//  U-plane if output data format is YUV planar.
//  output data format is YUV planar.
//  Due to clock gating, the primary
//  OUTPUT_TO_MEMORY must be enabled and the
//  OUTPUT_FORMAT must be set to a planar format
//  prior to writing this register
#define LWE24D_VI_VB0_START_ADDRESS_U_0_VB0_START_ADDRESS_U                      31:0


//(linked to First Output)
#define LWE24D_VI_VB0_BASE_ADDRESS_U_0                 (0x34)
//  This is the first byte address of video
//  buffer 0 U-plane if output data format
//  is planar.
#define LWE24D_VI_VB0_BASE_ADDRESS_U_0_VB0_BASE_ADDRESS_U                        31:0


#define LWE24D_VI_VB0_START_ADDRESS_V_0                        (0x35)
//  This is byte address of video buffer 0
//  V-plane if output data format is YUV planar.
//  output data format is YUV planar.
//  Due to clock gating, the primary
//  OUTPUT_TO_MEMORY must be enabled and the
//  OUTPUT_FORMAT must be set to a planar format
//  prior to writing this register
#define LWE24D_VI_VB0_START_ADDRESS_V_0_VB0_START_ADDRESS_V                      31:0


//(linked to First Output)
#define LWE24D_VI_VB0_BASE_ADDRESS_V_0                 (0x36)
//  This is byte address of video buffer 0
//  V-plane if output data format is YUV planar.
//  output data format is YUV planar.
#define LWE24D_VI_VB0_BASE_ADDRESS_V_0_VB0_BASE_ADDRESS_V                        31:0


#define LWE24D_VI_VB0_SCRATCH_ADDRESS_UV_0                     (0x37)
//  If OUTPUT_FORMAT is YUV420PA, this is used.
//  This is byte address of video buffer 0
//  UV intermediate data is saved here during the
//  YUV422 to YUV420PA colwersion.
//  The size allocated needs to match the
//  FIRST_FRAME_WIDTH register setting
#define LWE24D_VI_VB0_SCRATCH_ADDRESS_UV_0_VB0_SCRATCH_ADDRESS_UV                        31:0


// This is the size of the frame being written to memory.
// Apply decimation or averaging to callwlate the output frame
// size.  Whether or not downscaling is used specify whatever the
// size of the frame being written to memory.
#define LWE24D_VI_FIRST_OUTPUT_FRAME_SIZE_0                    (0x38)
// frame width in pixel which VI needs to process
#define LWE24D_VI_FIRST_OUTPUT_FRAME_SIZE_0_FIRST_FRAME_WIDTH                    12:0

// frame height in lines which VI needs to process
#define LWE24D_VI_FIRST_OUTPUT_FRAME_SIZE_0_FIRST_FRAME_HEIGHT                   28:16


#define LWE24D_VI_VB0_COUNT_FIRST_0                    (0x39)
// Video Buffer Set 0 Count
//  This specifies the number of buffers in
//  video buffer set 0.
#define LWE24D_VI_VB0_COUNT_FIRST_0_VB0_COUNT_1                  7:0


#define LWE24D_VI_VB0_SIZE_FIRST_0                     (0x3a)
// Video Buffer Set 0 Horizontal Size
//  This parameter specifies the line stride
//  (in pixels) for lines in the video buffer
//  set 0.
//  For YUV non-planar format, this parameter
//  must be programmed as multiple of 2 pixels
//  (bit 0 is ignored).
//  For YUV planar format, this parameter
//  must be programmed as multiple of 8 pixels
//  (bits 2-0 are ignored) and it specifies the
//  luma line stride or twice the chroma line
//  stride.
//  This value will be divided by 2 for chroma
//  buffers for YUV422 and YUV420 planar formats
#define LWE24D_VI_VB0_SIZE_FIRST_0_VB0_H_SIZE_1                  12:0

// Video Buffer Set 0 Vertical Size
//  This specifies the number of lines in each
//  buffer in video buffer set 0.
//  This value will be divided by 2 for chroma
//  buffers for YUV420 planar formats
#define LWE24D_VI_VB0_SIZE_FIRST_0_VB0_V_SIZE_1                  28:16


#define LWE24D_VI_VB0_BUFFER_STRIDE_FIRST_0                    (0x3b)
// Video Buffer Set 0 Luma Buffer Stride
//  This is luma buffer stride (in bytes)
#define LWE24D_VI_VB0_BUFFER_STRIDE_FIRST_0_VB0_BUFFER_STRIDE_L                  29:0

// Video Buffer Set 0 Chroma Buffer Stride   00= Equal to Luma Buffer Stride
//   01= Equal to Luma Buffer Stride divided by 2
//       in this case Luma Buffer Stride should
//       be multiple of 2 bytes.
//   10= Equal to Luma Buffer Stride divided by 4
//       in this case Luma Buffer Stride should
//       be multiple of 4 bytes.
//   1x= Reserved
#define LWE24D_VI_VB0_BUFFER_STRIDE_FIRST_0_VB0_BUFFER_STRIDE_C                  31:30
#define LWE24D_VI_VB0_BUFFER_STRIDE_FIRST_0_VB0_BUFFER_STRIDE_C_CBS1X                  (0)
#define LWE24D_VI_VB0_BUFFER_STRIDE_FIRST_0_VB0_BUFFER_STRIDE_C_CBS2X                  (1)
#define LWE24D_VI_VB0_BUFFER_STRIDE_FIRST_0_VB0_BUFFER_STRIDE_C_CBS4X                  (2)

// SECOND OUTPUT Registers
// These registers are used to setup the second of two memory outputs for VI
// Address; Frame size; Count; Size (line stride and block height); and Buffer Stride

#define LWE24D_VI_VB0_START_ADDRESS_SECOND_0                   (0x3c)
//  This is byte address of video buffer 0 if
//  output data format is RGB or YUV non-planar.
//  This is byte address of video buffer 0
//  This output data is read by the SB
#define LWE24D_VI_VB0_START_ADDRESS_SECOND_0_VB0_START_ADDRESS_2                 31:0


#define LWE24D_VI_VB0_BASE_ADDRESS_SECOND_0                    (0x3d)
//  This is byte address of video buffer 0 if
//  output data format is RGB or non-planar.
//  This is the first byte address of video
//  buffer
#define LWE24D_VI_VB0_BASE_ADDRESS_SECOND_0_VB0_BASE_ADDRESS_2                   31:0


#define LWE24D_VI_SECOND_OUTPUT_FRAME_SIZE_0                   (0x3e)
// frame width in pixel which VI needs to process
#define LWE24D_VI_SECOND_OUTPUT_FRAME_SIZE_0_SECOND_FRAME_WIDTH                  12:0

// frame height in lines which VI needs to process
#define LWE24D_VI_SECOND_OUTPUT_FRAME_SIZE_0_SECOND_FRAME_HEIGHT                 28:16


#define LWE24D_VI_VB0_COUNT_SECOND_0                   (0x3f)
//
//  This specifies the number of buffers in
//  video buffer set 0.
#define LWE24D_VI_VB0_COUNT_SECOND_0_VB0_COUNT_2                 7:0


#define LWE24D_VI_VB0_SIZE_SECOND_0                    (0x40)
// Video Buffer Set 0 Horizontal Size
//  This parameter specifies the line stride
//  (in pixels) for lines in the video buffer
//  set 0.
//  For YUV non-planar format, this parameter
//  must be programmed as multiple of 2 pixels
//  (bit 0 is ignored).
#define LWE24D_VI_VB0_SIZE_SECOND_0_VB0_H_SIZE_2                 12:0

// Video Buffer Set 0 Vertical Size
//  This specifies the number of lines in each
//  buffer in video buffer set 0.
#define LWE24D_VI_VB0_SIZE_SECOND_0_VB0_V_SIZE_2                 28:16


#define LWE24D_VI_VB0_BUFFER_STRIDE_SECOND_0                   (0x41)
// Video Buffer Set 0 Luma Buffer Stride
//  This is luma buffer stride (in bytes)
#define LWE24D_VI_VB0_BUFFER_STRIDE_SECOND_0_VB0_BUFFER_STRIDE_2                 29:0

// This register controls horizontal low-pass filtering which can be enabled to improve quality
// of the decimated image. The only valid programming values for this register are:
//   0x000000002400240          No filtering
//   0x00000000DBE092E          1-HPF^3
//   0x000000001B60126          1-HPF^2
//   0x000000005B70127          (1-HPF^2+LPF)/2
//   0x000000006480248          LPF
//   0x000000004910001          (LPF+LPF^2)/2
//   0x000000000900000          LPF^2
//   0x000000004980008          LPF^3
//   0x000000007980308          LPF^2 * (0.5,0,0.5)
//   0x000000007f80368          LPF * (0.5,0,0.5) * (2,-3,2)
// The above list is ordered from the widest band-pass filter to the narrowest band-pass filter.
#define LWE24D_VI_H_LPF_NO_FILTER      576
#define LWE24D_VI_H_LPF_ONE_MINUS_HPF_LWBED_C  3518
#define LWE24D_VI_H_LPF_ONE_MINUS_HPF_LWBED_L  2350
#define LWE24D_VI_H_LPF_ONE_MINUS_HPF_SQUARED_C        438
#define LWE24D_VI_H_LPF_ONE_MINUS_HPF_SQUARED_L        294
#define LWE24D_VI_H_LPF_ONE_MINUS_HPF_SQUARED_PLUS_LPF_C       1463
#define LWE24D_VI_H_LPF_ONE_MINUS_HPF_SQUARED_PLUS_LPF_L       295
#define LWE24D_VI_H_LPF_LPF_C  1608
#define LWE24D_VI_H_LPF_LPF_L  584
#define LWE24D_VI_H_LPF_LPF_PLUS_LPF_SQUARED_C 1169
#define LWE24D_VI_H_LPF_LPF_PLUS_LPF_SQUARED_L 1
#define LWE24D_VI_H_LPF_LPF_SQUARED_C  144
#define LWE24D_VI_H_LPF_LPF_SQUARED_L  0
#define LWE24D_VI_H_LPF_LPF_LWBED_C    1176
#define LWE24D_VI_H_LPF_LPF_LWBED_L    8
#define LWE24D_VI_H_LPF_LPF_SQUARED_SCALED_C   1944
#define LWE24D_VI_H_LPF_LPF_SQUARED_SCALED_L   776
#define LWE24D_VI_H_LPF_LPF_SQUARED_SCALED2_C  2040
#define LWE24D_VI_H_LPF_LPF_SQUARED_SCALED2_L  872

#define LWE24D_VI_H_LPF_CONTROL_0                      (0x42)
// Horizontal LPF Luminance filter
//  This controls low pass filter for Y data.
#define LWE24D_VI_H_LPF_CONTROL_0_H_LPF_L                        12:0

// Horizontal LPF Chrominance filter
//  This controls low pass filter for U V data.
#define LWE24D_VI_H_LPF_CONTROL_0_H_LPF_C                        28:16

// Horizontal pixel processing starts with horizontal low-pass filtering.
// Following horizontal low-pass filtering, horizontal down-scaling (decimation) can then be
// performed with or without horizontal averaging.
// If horizontal down-scaling (decimation) is performed without horizontal averaging, the
// down-scaling factor is specified by input active period and output frame size.  Because the
// VI has two input methods (VIP and HOST) and two memory outputs, there are mux selects to indicate
// which registers to use in callwlating the input and output frame sizes.
// If horizontal down-scaling is performed with horizontal averaging, the down-scaling factors
// are limited to few factors determined by H_AVG_CONTROL.  When enabling averaging PLEASE be careful
// that the input and output ratios match the formula for the averaging decimation ratio exactly to the
// pixel/line. The formula for each of the Averaging Decimation Ratio is as follows:
//
// Averaging Decimation Formalae
// x = input size
// y(x) = output size
// 2-pixel averaging and 1/2 downscaling: y(x) = Floor(x/2)
// 4-pixel averaging and 1/3 downscaling: y(x) = Floor((x-1)/3)
// 4-pixel averaging and 1/4 downscaling: y(x) = Floor(x/4)
// 8-pixel averaging and 1/7 downscaling: y(x) = Floor((x-1)/7)
// 8-pixel averaging and 1/8 downscaling: y(x) = Floor(x/8)
//
// Horizontal Decimation Algorithm:
// The Horizontal Decimator decides which pixels to drop by using a simple DDA algorithm.
// The aclwmulator will continue to add the value of the output width (numerator) for each
// pixel until the sum is equal or greater than the input width (denominator).  When the sum
// is greater or equal to the input width (denominator), the hardware will flag that pixel as
// a pixel to be written out to memory.  At the same time, the input width (denominator) will
// be subtracted from the sum and the difference will be loaded back into the aclwmulator for
// the next line.  By default the aclwmulator is initialized with 0's upon reset.  However the
// user can set the H_DEC_INIT_VAL to initialize the aclwmulator with a certain value from
// 0 to the input width (denominator).  Any H_DEC_INIT_VAL that is greater or equal to the
// difference of the input width (denominator) and the output width (numerator) will cause the
// first pixel to be written out to memory.  This register shifts the phase of the decimation
// pattern.

#define LWE24D_VI_H_DOWNSCALE_CONTROL_0                        (0x43)
// Input Horizontal Size Select  Selects between the VIP and HOST input active
//  area widths for the denominator in the
//  downscaling ratio.  Uses VIP_H_ACTIVE_PERIOD or
//  HOST_H_ACTIVE_PERIOD, which is the width of the
//  data after cropping.  This is effective only when
//  H_AVERAGING is DISABLED and H_DOWNSCALING is
//  ENABLED.
#define LWE24D_VI_H_DOWNSCALE_CONTROL_0_INPUT_H_SIZE_SEL                 0:0
#define LWE24D_VI_H_DOWNSCALE_CONTROL_0_INPUT_H_SIZE_SEL_VIP                   (0)
#define LWE24D_VI_H_DOWNSCALE_CONTROL_0_INPUT_H_SIZE_SEL_HOST                  (1)

// Output Horizontal Size Select  Selects between the first and second memory output
//  frame widths for the numerator in the downscaling
//  ratio.  Uses FIRST_FRAME_WIDTH or
//  SECOND_FRAME_WIDTH.
//  This is effective
//  only when H_AVERAGING is DISABLED and
//  H_DOWNSCALING is ENABLED.
#define LWE24D_VI_H_DOWNSCALE_CONTROL_0_OUTPUT_H_SIZE_SEL                        1:1
#define LWE24D_VI_H_DOWNSCALE_CONTROL_0_OUTPUT_H_SIZE_SEL_FIRST                        (0)
#define LWE24D_VI_H_DOWNSCALE_CONTROL_0_OUTPUT_H_SIZE_SEL_SECOND                       (1)

//  Selects input horizontal size into scalers (extension field)
//  00= Hor. size selected with INPUT_H_SIZE_SEL field (backward compatible)
//  01= Hor. size of CSI_PPA is provided by CSI_PPA_H_ACTIVE register
//  10= Hor. size of CSI_PPB is provided by CSI_PPB_H_ACTIVE register
//  11= Hor. size of ISP     is provided by ISP_H_ACTIVE     register
// 7:4 reserved
#define LWE24D_VI_H_DOWNSCALE_CONTROL_0_INPUT_H_SIZE_SEL_EXT                     3:2
#define LWE24D_VI_H_DOWNSCALE_CONTROL_0_INPUT_H_SIZE_SEL_EXT_USE_INPUT_H_SIZE_SEL                      (0)
#define LWE24D_VI_H_DOWNSCALE_CONTROL_0_INPUT_H_SIZE_SEL_EXT_CSI_PPA                   (1)
#define LWE24D_VI_H_DOWNSCALE_CONTROL_0_INPUT_H_SIZE_SEL_EXT_CSI_PPB                   (2)
#define LWE24D_VI_H_DOWNSCALE_CONTROL_0_INPUT_H_SIZE_SEL_EXT_ISP                       (3)

// Horizontal Averaging Control  This specifies the number of pixels to
//  average and to decimate horizontally.
//   000= 2-pixel averaging and 1/2 down-scaling
//   001= 4-pixel averaging and 1/3 down-scaling
//   010= 4-pixel averaging and 1/4 down-scaling
//   011= 8-pixel averaging and 1/7 down-scaling
//   100= 8-pixel averaging and 1/8 down-scaling
//   other= reserved
#define LWE24D_VI_H_DOWNSCALE_CONTROL_0_H_AVG_CONTROL                    10:8
#define LWE24D_VI_H_DOWNSCALE_CONTROL_0_H_AVG_CONTROL_A2D2                     (0)
#define LWE24D_VI_H_DOWNSCALE_CONTROL_0_H_AVG_CONTROL_A4D3                     (1)
#define LWE24D_VI_H_DOWNSCALE_CONTROL_0_H_AVG_CONTROL_A4D4                     (2)
#define LWE24D_VI_H_DOWNSCALE_CONTROL_0_H_AVG_CONTROL_A8D7                     (3)
#define LWE24D_VI_H_DOWNSCALE_CONTROL_0_H_AVG_CONTROL_A8D8                     (4)

// Horizontal Decimation Aclwmulator Initial Value
//  The user may initialized the H-Dec aclwmulator with
//  a value between 0-(H_ACTIVE_PERIOD) to change the phase
//  of the decimation pattern.  This will allow the user
//  to decide which is the first pixel to keep.
#define LWE24D_VI_H_DOWNSCALE_CONTROL_0_H_DEC_INIT_VAL                   28:16

// Vertical processing consists of optional vertical down-scaling (decimation) which can be
// performed with or without vertical averaging.
// If vertical down-scaling (decimation) is performed without vertical averaging, the
// down-scaling factor is specified by input active period and output frame size.  Because the
// VI has two input methods (VIP and HOST) and two memory outputs, there are mux selects to indicate
// which registers to use in callwlating the input and output frame sizes.
// If horizontal down-scaling is performed with vertical averaging, the down-scaling factors
// are limited to few factors determined by V_AVG_CONTROL.  When enabling averaging PLEASE be careful
// that the input and output ratios match the formula for the averaging decimation ratio exactly to the
// pixel/line. The formula for each of the Averaging Decimation Ratio is as follows:
//
// Averaging Decimation Formalae
// x = input size
// y(x) = output size
// 2-pixel averaging and 1/2 downscaling: y(x) = Floor(x/2)
// 4-pixel averaging and 1/3 downscaling: y(x) = Floor((x-1)/3)
// 4-pixel averaging and 1/4 downscaling: y(x) = Floor(x/4)
// 8-pixel averaging and 1/7 downscaling: y(x) = Floor((x-1)/7)
// 8-pixel averaging and 1/8 downscaling: y(x) = Floor(x/8)
//
// Vertical Decimation Algorithm: (same as the Horizontal Decimation Algorithm)
// The Vertical Decimator decides which pixels to drop by using a simple DDA algorithm.
// The aclwmulator will continue to add the value of the output height (numerator) for each
// line until the sum is equal or greater than the input height (denominator).  When the sum
// is greater or equal to the input height (denominator), the hardware will flag that line as
// a line to be written out to memory.  At the same time, the input height (denominator) will
// be subtracted from the sum and the difference will be loaded back into the aclwmulator for
// the next line.  By default the aclwmulator is initialized with 0's upon reset.  However the
// user can set the V_DEC_INIT_VAL to initialize the aclwmulator with a certain value from
// 0 to the input height (denominator).  Any V_DEC_INIT_VAL that is greater or equal to the
// difference of the input height (denominator) and the output height (numerator) will cause the
// first line to be written out to memory.  This register shifts the phase of the decimation
// pattern.

#define LWE24D_VI_V_DOWNSCALE_CONTROL_0                        (0x44)
// Input Vertical Size Select  Selects between the VIP and HOST input active
//  area heights for the denominator in the
//  downscaling ratio.  Uses VIP_V_ACTIVE_PERIOD or
//  HOST_V_ACTIVE_PERIOD, which is the height of the
//  data after cropping.  This is effective only when
//  V_AVERAGING is DISABLED and V_DOWNSCALING is
//  ENABLED.
#define LWE24D_VI_V_DOWNSCALE_CONTROL_0_INPUT_V_SIZE_SEL                 0:0
#define LWE24D_VI_V_DOWNSCALE_CONTROL_0_INPUT_V_SIZE_SEL_VIP                   (0)
#define LWE24D_VI_V_DOWNSCALE_CONTROL_0_INPUT_V_SIZE_SEL_HOST                  (1)

// Output Vertical Size Select  Selects between the first and second memory output
//  frame heights for the numerator in the downscaling
//  ratio.  Uses FIRST_FRAME_HEIGHT or
//  SECOND_FRAME_HEIGHT.
//  This is effective
//  only when V_AVERAGING is DISABLED and
//  V_DOWNSCALING is ENABLED.
#define LWE24D_VI_V_DOWNSCALE_CONTROL_0_OUTPUT_V_SIZE_SEL                        1:1
#define LWE24D_VI_V_DOWNSCALE_CONTROL_0_OUTPUT_V_SIZE_SEL_FIRST                        (0)
#define LWE24D_VI_V_DOWNSCALE_CONTROL_0_OUTPUT_V_SIZE_SEL_SECOND                       (1)

//  Selects input vertical size into scalers (extension field)
//  00= Vert. size selected with INPUT_V_SIZE_SEL field (backward compatible)
//  01= Vert. size of CSI_PPA is provided by CSI_PPA_V_ACTIVE register
//  10= Vert. size of CSI_PPB is provided by CSI_PPB_V_ACTIVE register
//  11= Vert. size of ISP     is provided by ISP_V_ACTIVE     register
// 7:4 reserved
#define LWE24D_VI_V_DOWNSCALE_CONTROL_0_INPUT_V_SIZE_SEL_EXT                     3:2
#define LWE24D_VI_V_DOWNSCALE_CONTROL_0_INPUT_V_SIZE_SEL_EXT_USE_INPUT_V_SIZE_SEL                      (0)
#define LWE24D_VI_V_DOWNSCALE_CONTROL_0_INPUT_V_SIZE_SEL_EXT_CSI_PPA                   (1)
#define LWE24D_VI_V_DOWNSCALE_CONTROL_0_INPUT_V_SIZE_SEL_EXT_CSI_PPB                   (2)
#define LWE24D_VI_V_DOWNSCALE_CONTROL_0_INPUT_V_SIZE_SEL_EXT_ISP                       (3)

// Vertical Averaging Control  This specifies the number of lines to
//  average and to decimate vertically.
//   000= 2-line averaging and 1/2 down-scaling
//   001= 4-line averaging and 1/3 down-scaling
//   010= 4-line averaging and 1/4 down-scaling
//   011= 8-line averaging and 1/7 down-scaling
//   100= 8-line averaging and 1/8 down-scaling
//   other= reserved
#define LWE24D_VI_V_DOWNSCALE_CONTROL_0_V_AVG_CONTROL                    10:8
#define LWE24D_VI_V_DOWNSCALE_CONTROL_0_V_AVG_CONTROL_A2D2                     (0)
#define LWE24D_VI_V_DOWNSCALE_CONTROL_0_V_AVG_CONTROL_A4D3                     (1)
#define LWE24D_VI_V_DOWNSCALE_CONTROL_0_V_AVG_CONTROL_A4D4                     (2)
#define LWE24D_VI_V_DOWNSCALE_CONTROL_0_V_AVG_CONTROL_A8D7                     (3)
#define LWE24D_VI_V_DOWNSCALE_CONTROL_0_V_AVG_CONTROL_A8D8                     (4)

// Flexible Vertical Scaling   0 = disabled, V_AVG_CONTROL specifies both
//       vertical averaging and down-scaling
//       factor.
//   1 = enabled, fixed 2-line averaging with
//       vertical downscaling controlled by
//       V_DOWN_N and V_DOWN_D.
#define LWE24D_VI_V_DOWNSCALE_CONTROL_0_FLEXIBLE_VSCALE                  12:12
#define LWE24D_VI_V_DOWNSCALE_CONTROL_0_FLEXIBLE_VSCALE_DISABLED                       (0)
#define LWE24D_VI_V_DOWNSCALE_CONTROL_0_FLEXIBLE_VSCALE_ENABLED                        (1)

// Multi-Tap Vertical Averaging Filter   0 = disabled
//   1 = enabled
//  This will enable the Multi-Tap filtering
//  when the Vertical Averaging is enabled.
//  The filter settings will depend on the
//  V_AVG_CONTROL value.
//  000 - 3 Taps (1,2,1)/4
//  001 - 5 Taps (1,2,2,2,1)/8
//  010 - 6 Taps (1,1,2,2,1,1)/8
//  011 - 11 Taps (1,1,1,2,2,2,2,2,1,1,1)/16
//  100 - 12 Taps (1,1,1,1,2,2,2,2,1,1,1,1)/16
#define LWE24D_VI_V_DOWNSCALE_CONTROL_0_MULTI_TAP_V_AVG_FILTER                   13:13
#define LWE24D_VI_V_DOWNSCALE_CONTROL_0_MULTI_TAP_V_AVG_FILTER_DISABLED                        (0)
#define LWE24D_VI_V_DOWNSCALE_CONTROL_0_MULTI_TAP_V_AVG_FILTER_ENABLED                 (1)

// Vertical Decimation Aclwmulator Initial Value
//  The user may initialized the V-Dec aclwmulator with
//  a value between 0-(V_ACTIVE_PERIOD) to change the phase
//  of the decimation pattern.  This will allow the user
//  to decide which is the first line to keep.
#define LWE24D_VI_V_DOWNSCALE_CONTROL_0_V_DEC_INIT_VAL                   28:16

// Specifies whether odd/even field affects vertical  decimation.
//   0 = disabled - odd/even field affects the vertical downscaling
//   1 = enabled - field is ignored in vertical downscaling
#define LWE24D_VI_V_DOWNSCALE_CONTROL_0_IGNORE_FIELD                     28:28
#define LWE24D_VI_V_DOWNSCALE_CONTROL_0_IGNORE_FIELD_DISABLED                  (0)
#define LWE24D_VI_V_DOWNSCALE_CONTROL_0_IGNORE_FIELD_ENABLED                   (1)

// Color Space Colwersion coefficients.
// The CSC can be used for YUV to RGB colwersion with brightness and hue/saturation control.
// For Y color, the Y offset is applied first and saturation (clipping) is performed
//   immediately after the Y offset is applied.
//   R = sat(KYRGB * sat(Y + YOF) + KUR * U + KVR * V)
//   G = sat(KYRGB * sat(Y + YOF) + KUG * U + KVG * V)
//   B = sat(KYRGB * sat(Y + YOF) + KUB * U + KVB * V)
// Saturation and rounding is performed in the range of 0 to 255 for the above equations.
//
// Typical values are:
//   YOF = -16.000, KYRGB =  1.1644
//   KUR =  0.0000, KVR   = -1.5960
//   KUG = -0.3918, KVG   = -0.8130
//   KUB =  2.0172, KVB   =  0.0000
//   KUR and KVB are typically 0.0000 but they may be programmed non-zero for hue rotation.
//
// The CSC can also take RGB input, in which case YOF, KVB, KUG, KUR should be programmed to 0
//   and KYRGB will be forced to 0 by the hardware for generating R and B. KYRGB will not be
//   forced to 0 for generating G. KVR, KYRGB, and KUB can be programmed to 1.0 or used as
//   gain control for R, G, B correspondingly.
// Note that color value ranges from 0 to 255 for Y, R, G, B and -128 to 127 for U and V.

#define LWE24D_VI_CSC_Y_0                      (0x45)
// Y Offset in s.7.0 format
#define LWE24D_VI_CSC_Y_0_YOF                    7:0

// Y Gain for R, G, B colors in 2.8 format
#define LWE24D_VI_CSC_Y_0_KYRGB                  25:16


#define LWE24D_VI_CSC_UV_R_0                   (0x46)
// U coefficients for R in s.2.8 format
#define LWE24D_VI_CSC_UV_R_0_KUR                 10:0

// V coefficients for R in s.2.8 format
#define LWE24D_VI_CSC_UV_R_0_KVR                 26:16


#define LWE24D_VI_CSC_UV_G_0                   (0x47)
// U coefficients for G in s.1.8 format
#define LWE24D_VI_CSC_UV_G_0_KUG                 9:0

// V coefficients for G in s.1.8 format
#define LWE24D_VI_CSC_UV_G_0_KVG                 25:16


#define LWE24D_VI_CSC_UV_B_0                   (0x48)
// U coefficients for B in s.2.8 format
#define LWE24D_VI_CSC_UV_B_0_KUB                 10:0

// V coefficients for B in s.2.8 format
#define LWE24D_VI_CSC_UV_B_0_KVB                 26:16


#define LWE24D_VI_CSC_ALPHA_0                  (0x49)
// When output format to memory is selected
//  for RGB888, the pixel data is 32-bit aligned
//  The value programmed here will be appended to the
//  RGB888 data as the 8 MSBs and can be used as an
//  alpha value.
#define LWE24D_VI_CSC_ALPHA_0_RGB888_ALPHA                       7:0


#define LWE24D_VI_HOST_VSYNC_0                 (0x4a)
// This triggers VI's internal VSYNC generation
// Always write once to this register with '1'
// before writing the Frame's data to Y_FIFO_DATA
#define LWE24D_VI_HOST_VSYNC_0_HOST_VSYNC_TRIGGER                        0:0

// **** This eventually needs to be moved to command buffer interface.
// This register is used initialize VI module when INPUT_SOURCE is HOST.
// **** This register has a dual use purpose.  Host input VSYNC is created by
// writing to this register.

#define LWE24D_VI_COMMAND_0                    (0x4b)
// Process Odd/Even field  (effective when INPUT_SOURCE is HOST)
//  Writing to this bit will initialize VI
//  to receive one field of video.
//   0= odd field
//   1= even field
#define LWE24D_VI_COMMAND_0_PROCESS_FIELD                        0:0
#define LWE24D_VI_COMMAND_0_PROCESS_FIELD_ODD                  (0)
#define LWE24D_VI_COMMAND_0_PROCESS_FIELD_EVEN                 (1)

// Y-FIFO Threshold
//  This specifies maximum number of filled
//  locations in Y-FIFO for the Y-FIFO Threshold
//  Status bit.
#define LWE24D_VI_COMMAND_0_Y_FIFO_THRESHOLD                     11:8

// Vertical Counter Threshold
//  This specifies a threshold which, when
//  exceeded, would generate the vertical
//  counter interrupt if the interrupt is
//  enabled. This is used to detect the case
//  when the host is sending too many input data
//  than expected by VI module.
#define LWE24D_VI_COMMAND_0_V_COUNTER_THRESHOLD                  28:16

// **** This is not needed if host input video goes through command buffer interface.

#define LWE24D_VI_HOST_FIFO_STATUS_0                   (0x4c)
// This indicates the number of filled locations
//  in Y-FIFO. If the returned value is 4'h0, the
//  fifo is empty and if the returned value is
//  4'hF then the fifo is full.
#define LWE24D_VI_HOST_FIFO_STATUS_0_Y_FIFO_STATUS                       3:0

// This indicates the number of filled locations
//  in U-FIFO. If the returned value is 3'h0, the
//  fifo is empty and if the returned value is
//  3'h7 then the fifo is full.
#define LWE24D_VI_HOST_FIFO_STATUS_0_U_FIFO_STATUS                       10:8

// This indicates the number of filled locations
//  in V-FIFO. If the returned value is 3'h0, the
//  fifo is empty and if the returned value is
//  3'h7 then the fifo is full.
#define LWE24D_VI_HOST_FIFO_STATUS_0_V_FIFO_STATUS                       14:12


#define LWE24D_VI_INTERRUPT_MASK_0                     (0x4d)
// VD8 pin Interrupt Mask  This bit controls interrupt when VD8
//  rising/falling edge is detected.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_INTERRUPT_MASK_0_VD8_INT_MASK                  0:0
#define LWE24D_VI_INTERRUPT_MASK_0_VD8_INT_MASK_DISABLED                       (0)
#define LWE24D_VI_INTERRUPT_MASK_0_VD8_INT_MASK_ENABLED                        (1)

// VD9 pin Interrupt Mask  This bit controls interrupt when VD9
//  rising/falling edge is detected.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_INTERRUPT_MASK_0_VD9_INT_MASK                  1:1
#define LWE24D_VI_INTERRUPT_MASK_0_VD9_INT_MASK_DISABLED                       (0)
#define LWE24D_VI_INTERRUPT_MASK_0_VD9_INT_MASK_ENABLED                        (1)

// VD10 pin Interrupt Mask  This bit controls interrupt when VD10
//  rising/falling edge is detected.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_INTERRUPT_MASK_0_VD10_INT_MASK                 2:2
#define LWE24D_VI_INTERRUPT_MASK_0_VD10_INT_MASK_DISABLED                      (0)
#define LWE24D_VI_INTERRUPT_MASK_0_VD10_INT_MASK_ENABLED                       (1)

// VD11 pin Interrupt Mask  This bit controls interrupt when VD11
//  rising/falling edge is detected.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_INTERRUPT_MASK_0_VD11_INT_MASK                 3:3
#define LWE24D_VI_INTERRUPT_MASK_0_VD11_INT_MASK_DISABLED                      (0)
#define LWE24D_VI_INTERRUPT_MASK_0_VD11_INT_MASK_ENABLED                       (1)

// VGP4 pin Interrupt Mask  This bit controls interrupt when VGP4
//  rising/falling edge is detected.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_INTERRUPT_MASK_0_VGP4_INT_MASK                 4:4
#define LWE24D_VI_INTERRUPT_MASK_0_VGP4_INT_MASK_DISABLED                      (0)
#define LWE24D_VI_INTERRUPT_MASK_0_VGP4_INT_MASK_ENABLED                       (1)

// VGP5 pin Interrupt Mask  This bit controls interrupt when VGP5
//  rising/falling edge is detected.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_INTERRUPT_MASK_0_VGP5_INT_MASK                 5:5
#define LWE24D_VI_INTERRUPT_MASK_0_VGP5_INT_MASK_DISABLED                      (0)
#define LWE24D_VI_INTERRUPT_MASK_0_VGP5_INT_MASK_ENABLED                       (1)

// VGP6 pin Interrupt Mask  This bit controls interrupt when VGP6
//  rising/falling edge is detected.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_INTERRUPT_MASK_0_VGP6_INT_MASK                 6:6
#define LWE24D_VI_INTERRUPT_MASK_0_VGP6_INT_MASK_DISABLED                      (0)
#define LWE24D_VI_INTERRUPT_MASK_0_VGP6_INT_MASK_ENABLED                       (1)

// VHS pin Interrupt Mask  This bit controls interrupt when VHS
//  rising/falling edge is detected.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_INTERRUPT_MASK_0_VHS_INT_MASK                  7:7
#define LWE24D_VI_INTERRUPT_MASK_0_VHS_INT_MASK_DISABLED                       (0)
#define LWE24D_VI_INTERRUPT_MASK_0_VHS_INT_MASK_ENABLED                        (1)

// VVS pin Interrupt Mask  This bit controls interrupt when VVS
//  rising/falling edge is detected.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_INTERRUPT_MASK_0_VVS_INT_MASK                  8:8
#define LWE24D_VI_INTERRUPT_MASK_0_VVS_INT_MASK_DISABLED                       (0)
#define LWE24D_VI_INTERRUPT_MASK_0_VVS_INT_MASK_ENABLED                        (1)

// Vertical Counter Interrupt Mask  (effective when VIDEO_SOURCE is HOST)
//  This bit controls interrupt when the
//  vertical counter threshold is reached.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_INTERRUPT_MASK_0_V_COUNTER_INT_MASK                    9:9
#define LWE24D_VI_INTERRUPT_MASK_0_V_COUNTER_INT_MASK_DISABLED                 (0)
#define LWE24D_VI_INTERRUPT_MASK_0_V_COUNTER_INT_MASK_ENABLED                  (1)

// Y-FIFO Threshold Interrupt Mask  This bit controls interrupt when the number
//  of filled locations in Y-FIFO is equal or
//  greater than the Y_FIFO_THRESHOLD value.
//  This bit should be set to 1 only when
//  INPUT_SOURCE is HOST.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_INTERRUPT_MASK_0_Y_THRESHOLD_INT_MASK                  10:10
#define LWE24D_VI_INTERRUPT_MASK_0_Y_THRESHOLD_INT_MASK_DISABLED                       (0)
#define LWE24D_VI_INTERRUPT_MASK_0_Y_THRESHOLD_INT_MASK_ENABLED                        (1)

// Buffer Done First Output Interrupt Mask  This bit controls interrupt when the
//  First Output to memory has written
//  a buffer to memory.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_INTERRUPT_MASK_0_BUFFER_FIRST_OUTPUT_INT_MASK                  11:11
#define LWE24D_VI_INTERRUPT_MASK_0_BUFFER_FIRST_OUTPUT_INT_MASK_DISABLED                       (0)
#define LWE24D_VI_INTERRUPT_MASK_0_BUFFER_FIRST_OUTPUT_INT_MASK_ENABLED                        (1)

// Frame Done First Output Interrupt Mask  This bit controls interrupt when the
//  First Output to memory has written
//  a frame to memory.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_INTERRUPT_MASK_0_FRAME_FIRST_OUTPUT_INT_MASK                   12:12
#define LWE24D_VI_INTERRUPT_MASK_0_FRAME_FIRST_OUTPUT_INT_MASK_DISABLED                        (0)
#define LWE24D_VI_INTERRUPT_MASK_0_FRAME_FIRST_OUTPUT_INT_MASK_ENABLED                 (1)

// Buffer Done Second Output Interrupt Mask  This bit controls interrupt when the
//  Second Output to memory has written
//  a buffer to memory.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_INTERRUPT_MASK_0_BUFFER_SECOND_OUTPUT_INT_MASK                 13:13
#define LWE24D_VI_INTERRUPT_MASK_0_BUFFER_SECOND_OUTPUT_INT_MASK_DISABLED                      (0)
#define LWE24D_VI_INTERRUPT_MASK_0_BUFFER_SECOND_OUTPUT_INT_MASK_ENABLED                       (1)

// Buffer Done Second Output Interrupt Mask  This bit controls interrupt when the
//  Second Output to memory has written
//  a frame to memory.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_INTERRUPT_MASK_0_FRAME_SECOND_OUTPUT_INT_MASK                  14:14
#define LWE24D_VI_INTERRUPT_MASK_0_FRAME_SECOND_OUTPUT_INT_MASK_DISABLED                       (0)
#define LWE24D_VI_INTERRUPT_MASK_0_FRAME_SECOND_OUTPUT_INT_MASK_ENABLED                        (1)

// VI to EPP Error Interrupt Mask  This bit controls interrupt when the
//  VI drops data to the EPP because the
//  EPP is stalling the vi2epp bus and
//  data is coming from the pins
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_INTERRUPT_MASK_0_EPP_ERROR_INT_MASK                    15:15
#define LWE24D_VI_INTERRUPT_MASK_0_EPP_ERROR_INT_MASK_DISABLED                 (0)
#define LWE24D_VI_INTERRUPT_MASK_0_EPP_ERROR_INT_MASK_ENABLED                  (1)

// YUV420PA Error Interrupt Mask  This bit controls interrupt when the
//  VI does not average data because the
//  line buffer data is not ready from the
//  memory controller.  The VI will write
//  unaveraged data and will write the U,V
//  data from the even line in such cases.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_INTERRUPT_MASK_0_YUV420PA_ERROR_INT_MASK                       16:16
#define LWE24D_VI_INTERRUPT_MASK_0_YUV420PA_ERROR_INT_MASK_DISABLED                    (0)
#define LWE24D_VI_INTERRUPT_MASK_0_YUV420PA_ERROR_INT_MASK_ENABLED                     (1)

// VI to Peer stall - First Memory Output  This bit controls interrupt when the
//  VI drops peer bus packet(s) because the
//  peer is stalling the first output peer
//  bus and data is coming from the pins
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_INTERRUPT_MASK_0_FIRST_OUTPUT_PEER_STALL_INT_MASK                      17:17
#define LWE24D_VI_INTERRUPT_MASK_0_FIRST_OUTPUT_PEER_STALL_INT_MASK_DISABLED                   (0)
#define LWE24D_VI_INTERRUPT_MASK_0_FIRST_OUTPUT_PEER_STALL_INT_MASK_ENABLED                    (1)

// VI to Peer stall - Second Memory Output  This bit controls interrupt when the
//  VI drops peer bus packet(s) because the
//  peer is stalling the second output peer
//  bus and data is coming from the pins
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_INTERRUPT_MASK_0_SECOND_OUTPUT_PEER_STALL_INT_MASK                     18:18
#define LWE24D_VI_INTERRUPT_MASK_0_SECOND_OUTPUT_PEER_STALL_INT_MASK_DISABLED                  (0)
#define LWE24D_VI_INTERRUPT_MASK_0_SECOND_OUTPUT_PEER_STALL_INT_MASK_ENABLED                   (1)

// Write Buffer DMA to VI Stalls VI and causes an error  This bit controls interrupt when the
//  VI drops raw 8-bit stream data because
//  the Write Buffer DMA is stalling.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_INTERRUPT_MASK_0_DMA_STALL_INT_MASK                    19:19
#define LWE24D_VI_INTERRUPT_MASK_0_DMA_STALL_INT_MASK_DISABLED                 (0)
#define LWE24D_VI_INTERRUPT_MASK_0_DMA_STALL_INT_MASK_ENABLED                  (1)

// Stream 1 raise  This bit controls interrupt when the
//  the Stream 1 Raise is enabled and
//  returned
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_INTERRUPT_MASK_0_RAISE_STREAM_1_INT_MASK                       21:21
#define LWE24D_VI_INTERRUPT_MASK_0_RAISE_STREAM_1_INT_MASK_DISABLED                    (0)
#define LWE24D_VI_INTERRUPT_MASK_0_RAISE_STREAM_1_INT_MASK_ENABLED                     (1)

// Stream 2 raise  This bit controls interrupt when the
//  the Stream 2 Raise is enabled and
//  returned
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_INTERRUPT_MASK_0_RAISE_STREAM_2_INT_MASK                       22:22
#define LWE24D_VI_INTERRUPT_MASK_0_RAISE_STREAM_2_INT_MASK_DISABLED                    (0)
#define LWE24D_VI_INTERRUPT_MASK_0_RAISE_STREAM_2_INT_MASK_ENABLED                     (1)

// This bit controls interrupt when the
//  ISDB-T vi input gets an upstream error.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_INTERRUPT_MASK_0_TS_UPSTREAM_ERROR_INT_MASK                    23:23
#define LWE24D_VI_INTERRUPT_MASK_0_TS_UPSTREAM_ERROR_INT_MASK_DISABLED                 (0)
#define LWE24D_VI_INTERRUPT_MASK_0_TS_UPSTREAM_ERROR_INT_MASK_ENABLED                  (1)

// This bit controls interrupt when the
//  ISDB-T input get an underrun error
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_INTERRUPT_MASK_0_TS_UNDERRUN_ERROR_INT_MASK                    24:24
#define LWE24D_VI_INTERRUPT_MASK_0_TS_UNDERRUN_ERROR_INT_MASK_DISABLED                 (0)
#define LWE24D_VI_INTERRUPT_MASK_0_TS_UNDERRUN_ERROR_INT_MASK_ENABLED                  (1)

// This bit controls interrupt when the
//  ISDB-T input get an overrun error
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_INTERRUPT_MASK_0_TS_OVERRUN_ERROR_INT_MASK                     25:25
#define LWE24D_VI_INTERRUPT_MASK_0_TS_OVERRUN_ERROR_INT_MASK_DISABLED                  (0)
#define LWE24D_VI_INTERRUPT_MASK_0_TS_OVERRUN_ERROR_INT_MASK_ENABLED                   (1)

// This bit controls interrupt when the
//  ISDB-T input get a packet which means
//  FEC+BODY in totalsize but FEC and BODY
//  do not match FEC_SIZE and BODY_SIZE
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_INTERRUPT_MASK_0_TS_OTHER_PROTOCOL_ERROR_INT_MASK                      26:26
#define LWE24D_VI_INTERRUPT_MASK_0_TS_OTHER_PROTOCOL_ERROR_INT_MASK_DISABLED                   (0)
#define LWE24D_VI_INTERRUPT_MASK_0_TS_OTHER_PROTOCOL_ERROR_INT_MASK_ENABLED                    (1)

// This bit controls interrupt when VI drops
// data to MC.
#define LWE24D_VI_INTERRUPT_MASK_0_FIRST_OUTPUT_DROP_MC_DATA_INT_MASK                    27:27
#define LWE24D_VI_INTERRUPT_MASK_0_FIRST_OUTPUT_DROP_MC_DATA_INT_MASK_DISABLED                 (0)
#define LWE24D_VI_INTERRUPT_MASK_0_FIRST_OUTPUT_DROP_MC_DATA_INT_MASK_ENABLED                  (1)

// This bit controls interrupt when VI drops
// data to MC.
#define LWE24D_VI_INTERRUPT_MASK_0_SECOND_OUTPUT_DROP_MC_DATA_INT_MASK                   28:28
#define LWE24D_VI_INTERRUPT_MASK_0_SECOND_OUTPUT_DROP_MC_DATA_INT_MASK_DISABLED                        (0)
#define LWE24D_VI_INTERRUPT_MASK_0_SECOND_OUTPUT_DROP_MC_DATA_INT_MASK_ENABLED                 (1)


#define LWE24D_VI_INTERRUPT_TYPE_SELECT_0                      (0x4e)
// VD8 pin Interrupt Type  This bit controls interrupt VD8
//  if edge or level type
//   0= Edge type
//   1= Level type
#define LWE24D_VI_INTERRUPT_TYPE_SELECT_0_VD8_INT_TYPE                   0:0
#define LWE24D_VI_INTERRUPT_TYPE_SELECT_0_VD8_INT_TYPE_EDGE                    (0)
#define LWE24D_VI_INTERRUPT_TYPE_SELECT_0_VD8_INT_TYPE_LEVEL                   (1)

// VD9 pin Interrupt Type  This bit controls interrupt VD9
//   0= Edge type
//   1= Level type
#define LWE24D_VI_INTERRUPT_TYPE_SELECT_0_VD9_INT_TYPE                   1:1
#define LWE24D_VI_INTERRUPT_TYPE_SELECT_0_VD9_INT_TYPE_EDGE                    (0)
#define LWE24D_VI_INTERRUPT_TYPE_SELECT_0_VD9_INT_TYPE_LEVEL                   (1)

// VD10 pin Interrupt Type  This bit controls interrupt VD10
//   0= Edge type
//   1= Level type
#define LWE24D_VI_INTERRUPT_TYPE_SELECT_0_VD10_INT_TYPE                  2:2
#define LWE24D_VI_INTERRUPT_TYPE_SELECT_0_VD10_INT_TYPE_EDGE                   (0)
#define LWE24D_VI_INTERRUPT_TYPE_SELECT_0_VD10_INT_TYPE_LEVEL                  (1)

// VD11 pin Interrupt Type  This bit controls interrupt VD11
//   0= Edge type
//   1= Level type
#define LWE24D_VI_INTERRUPT_TYPE_SELECT_0_VD11_INT_TYPE                  3:3
#define LWE24D_VI_INTERRUPT_TYPE_SELECT_0_VD11_INT_TYPE_EDGE                   (0)
#define LWE24D_VI_INTERRUPT_TYPE_SELECT_0_VD11_INT_TYPE_LEVEL                  (1)

// VGP4 pin Interrupt Type  This bit controls interrupt VGP4
//   0= Edge type
//   1= Level type
#define LWE24D_VI_INTERRUPT_TYPE_SELECT_0_VGP4_INT_TYPE                  4:4
#define LWE24D_VI_INTERRUPT_TYPE_SELECT_0_VGP4_INT_TYPE_EDGE                   (0)
#define LWE24D_VI_INTERRUPT_TYPE_SELECT_0_VGP4_INT_TYPE_LEVEL                  (1)

// VGP5 pin Interrupt Type  This bit controls interrupt VGP5
//   0= Edge type
//   1= Level type
#define LWE24D_VI_INTERRUPT_TYPE_SELECT_0_VGP5_INT_TYPE                  5:5
#define LWE24D_VI_INTERRUPT_TYPE_SELECT_0_VGP5_INT_TYPE_EDGE                   (0)
#define LWE24D_VI_INTERRUPT_TYPE_SELECT_0_VGP5_INT_TYPE_LEVEL                  (1)

// VGP6 pin Interrupt Type  This bit controls interrupt VGP6
//   0= Edge type
//   1= Level type
#define LWE24D_VI_INTERRUPT_TYPE_SELECT_0_VGP6_INT_TYPE                  6:6
#define LWE24D_VI_INTERRUPT_TYPE_SELECT_0_VGP6_INT_TYPE_EDGE                   (0)
#define LWE24D_VI_INTERRUPT_TYPE_SELECT_0_VGP6_INT_TYPE_LEVEL                  (1)

// VHS pin Interrupt Type  This bit controls interrupt VHS
//   0= Edge type
//   1= Level type
#define LWE24D_VI_INTERRUPT_TYPE_SELECT_0_VHS_INT_TYPE                   7:7
#define LWE24D_VI_INTERRUPT_TYPE_SELECT_0_VHS_INT_TYPE_EDGE                    (0)
#define LWE24D_VI_INTERRUPT_TYPE_SELECT_0_VHS_INT_TYPE_LEVEL                   (1)

// VVS pin Interrupt Type  This bit controls interrupt VVS
//   0= Edge type
//   1= Level type
#define LWE24D_VI_INTERRUPT_TYPE_SELECT_0_VVS_INT_TYPE                   8:8
#define LWE24D_VI_INTERRUPT_TYPE_SELECT_0_VVS_INT_TYPE_EDGE                    (0)
#define LWE24D_VI_INTERRUPT_TYPE_SELECT_0_VVS_INT_TYPE_LEVEL                   (1)


#define LWE24D_VI_INTERRUPT_POLARITY_SELECT_0                  (0x4f)
// VD8 pin Interrupt Type  This bit controls interrupt VD8
//  if edge or level type
//   0= falling edge or low level
//   1= rising edge or high level
#define LWE24D_VI_INTERRUPT_POLARITY_SELECT_0_VD8_INT_POLARITY                   0:0
#define LWE24D_VI_INTERRUPT_POLARITY_SELECT_0_VD8_INT_POLARITY_LOW                     (0)
#define LWE24D_VI_INTERRUPT_POLARITY_SELECT_0_VD8_INT_POLARITY_HIGH                    (1)

// VD9 pin Interrupt Type  This bit controls interrupt VD9
//   0= falling edge or low level
//   1= rising edge or high level
#define LWE24D_VI_INTERRUPT_POLARITY_SELECT_0_VD9_INT_POLARITY                   1:1
#define LWE24D_VI_INTERRUPT_POLARITY_SELECT_0_VD9_INT_POLARITY_LOW                     (0)
#define LWE24D_VI_INTERRUPT_POLARITY_SELECT_0_VD9_INT_POLARITY_HIGH                    (1)

// VD10 pin Interrupt Type  This bit controls interrupt VD10
//   0= falling edge or low level
//   1= rising edge or high level
#define LWE24D_VI_INTERRUPT_POLARITY_SELECT_0_VD10_INT_POLARITY                  2:2
#define LWE24D_VI_INTERRUPT_POLARITY_SELECT_0_VD10_INT_POLARITY_LOW                    (0)
#define LWE24D_VI_INTERRUPT_POLARITY_SELECT_0_VD10_INT_POLARITY_HIGH                   (1)

// VD11 pin Interrupt Type  This bit controls interrupt VD11
//   0= falling edge or low level
//   1= rising edge or high level
#define LWE24D_VI_INTERRUPT_POLARITY_SELECT_0_VD11_INT_POLARITY                  3:3
#define LWE24D_VI_INTERRUPT_POLARITY_SELECT_0_VD11_INT_POLARITY_LOW                    (0)
#define LWE24D_VI_INTERRUPT_POLARITY_SELECT_0_VD11_INT_POLARITY_HIGH                   (1)

// VGP4 pin Interrupt Type  This bit controls interrupt VGP4
//   0= falling edge or low level
//   1= rising edge or high level
#define LWE24D_VI_INTERRUPT_POLARITY_SELECT_0_VGP4_INT_POLARITY                  4:4
#define LWE24D_VI_INTERRUPT_POLARITY_SELECT_0_VGP4_INT_POLARITY_LOW                    (0)
#define LWE24D_VI_INTERRUPT_POLARITY_SELECT_0_VGP4_INT_POLARITY_HIGH                   (1)

// VGP5 pin Interrupt Type  This bit controls interrupt VGP5
//   0= falling edge or low level
//   1= rising edge or high level
#define LWE24D_VI_INTERRUPT_POLARITY_SELECT_0_VGP5_INT_POLARITY                  5:5
#define LWE24D_VI_INTERRUPT_POLARITY_SELECT_0_VGP5_INT_POLARITY_LOW                    (0)
#define LWE24D_VI_INTERRUPT_POLARITY_SELECT_0_VGP5_INT_POLARITY_HIGH                   (1)

// VGP6 pin Interrupt Type  This bit controls interrupt VGP6
//   0= falling edge or low level
//   1= rising edge or high level
#define LWE24D_VI_INTERRUPT_POLARITY_SELECT_0_VGP6_INT_POLARITY                  6:6
#define LWE24D_VI_INTERRUPT_POLARITY_SELECT_0_VGP6_INT_POLARITY_LOW                    (0)
#define LWE24D_VI_INTERRUPT_POLARITY_SELECT_0_VGP6_INT_POLARITY_HIGH                   (1)

// VHS pin Interrupt Type  This bit controls interrupt VHS
//   0= falling edge or low level
//   1= rising edge or high level
#define LWE24D_VI_INTERRUPT_POLARITY_SELECT_0_VHS_INT_POLARITY                   7:7
#define LWE24D_VI_INTERRUPT_POLARITY_SELECT_0_VHS_INT_POLARITY_LOW                     (0)
#define LWE24D_VI_INTERRUPT_POLARITY_SELECT_0_VHS_INT_POLARITY_HIGH                    (1)

// VVS pin Interrupt Type  This bit controls interrupt VVS
//   0= falling edge or low level
//   1= rising edge or high level
#define LWE24D_VI_INTERRUPT_POLARITY_SELECT_0_VVS_INT_POLARITY                   8:8
#define LWE24D_VI_INTERRUPT_POLARITY_SELECT_0_VVS_INT_POLARITY_LOW                     (0)
#define LWE24D_VI_INTERRUPT_POLARITY_SELECT_0_VVS_INT_POLARITY_HIGH                    (1)

// This register returns interrupt status when read. Except for bits 15-14, when this register
// is written, the interrupt status corresponding to the bits written with 1 will be reset.
// Interrupt status corresponding to the bits written with 0 will be left unchanged.
// **** The following disclaimer is from SCx - not sure why they're needed ... interrupt should
//      not be generated when the corresponding interrupt enable bit is disabled.
// Note that interrupt status bits can be set even when their corresponding interrupt enable
// bits, in VI10R, are cleared. When these bits are set and their corresponding interrupt
// enable bits are set, an interrupt is generated. The interrupt can be cleared, or left
// unchanged, by writing 1, or 0, respectively to the corresponding bits in this register.
// Clearing the interrupt status bits does not affect the interrupt enable bits.

#define LWE24D_VI_INTERRUPT_STATUS_0                   (0x50)
// VD8 pin Interrupt Status  This bit controls interrupt when VD8
//  rising/falling edge is detected.
//   0= Interrupt not detected
//   1= Interrupt detected
#define LWE24D_VI_INTERRUPT_STATUS_0_VD8_INT_STATUS                      0:0
#define LWE24D_VI_INTERRUPT_STATUS_0_VD8_INT_STATUS_NOINTR                     (0)
#define LWE24D_VI_INTERRUPT_STATUS_0_VD8_INT_STATUS_INTR                       (1)

// VD9 pin Interrupt Status  This bit controls interrupt when VD9
//  rising/falling edge is detected.
//   0= Interrupt not detected
//   1= Interrupt detected
#define LWE24D_VI_INTERRUPT_STATUS_0_VD9_INT_STATUS                      1:1
#define LWE24D_VI_INTERRUPT_STATUS_0_VD9_INT_STATUS_NOINTR                     (0)
#define LWE24D_VI_INTERRUPT_STATUS_0_VD9_INT_STATUS_INTR                       (1)

// VD10 pin Interrupt Status  This bit controls interrupt when VD10
//  rising/falling edge is detected.
//   0= Interrupt not detected
//   1= Interrupt detected
#define LWE24D_VI_INTERRUPT_STATUS_0_VD10_INT_STATUS                     2:2
#define LWE24D_VI_INTERRUPT_STATUS_0_VD10_INT_STATUS_NOINTR                    (0)
#define LWE24D_VI_INTERRUPT_STATUS_0_VD10_INT_STATUS_INTR                      (1)

// VD11 pin Interrupt Status  This bit controls interrupt when VD11
//  rising/falling edge is detected.
//   0= Interrupt not detected
//   1= Interrupt detected
#define LWE24D_VI_INTERRUPT_STATUS_0_VD11_INT_STATUS                     3:3
#define LWE24D_VI_INTERRUPT_STATUS_0_VD11_INT_STATUS_NOINTR                    (0)
#define LWE24D_VI_INTERRUPT_STATUS_0_VD11_INT_STATUS_INTR                      (1)

// VGP4 pin Interrupt Status  This bit controls interrupt when VGP4
//  rising/falling edge is detected.
//   0= Interrupt not detected
//   1= Interrupt detected
#define LWE24D_VI_INTERRUPT_STATUS_0_VGP4_INT_STATUS                     4:4
#define LWE24D_VI_INTERRUPT_STATUS_0_VGP4_INT_STATUS_NOINTR                    (0)
#define LWE24D_VI_INTERRUPT_STATUS_0_VGP4_INT_STATUS_INTR                      (1)

// VGP5 pin Interrupt Status  This bit controls interrupt when VGP5
//  rising/falling edge is detected.
//   0= Interrupt not detected
//   1= Interrupt detected
#define LWE24D_VI_INTERRUPT_STATUS_0_VGP5_INT_STATUS                     5:5
#define LWE24D_VI_INTERRUPT_STATUS_0_VGP5_INT_STATUS_NOINTR                    (0)
#define LWE24D_VI_INTERRUPT_STATUS_0_VGP5_INT_STATUS_INTR                      (1)

// VGP6 pin Interrupt Status  This bit controls interrupt when VGP6
//  rising/falling edge is detected.
//   0= Interrupt not detected
//   1= Interrupt detected
#define LWE24D_VI_INTERRUPT_STATUS_0_VGP6_INT_STATUS                     6:6
#define LWE24D_VI_INTERRUPT_STATUS_0_VGP6_INT_STATUS_NOINTR                    (0)
#define LWE24D_VI_INTERRUPT_STATUS_0_VGP6_INT_STATUS_INTR                      (1)

// VHS pin Interrupt Status  This bit controls interrupt when VHS
//  rising/falling edge is detected.
//   0= Interrupt not detected
//   1= Interrupt detected
#define LWE24D_VI_INTERRUPT_STATUS_0_VHS_INT_STATUS                      7:7
#define LWE24D_VI_INTERRUPT_STATUS_0_VHS_INT_STATUS_NOINTR                     (0)
#define LWE24D_VI_INTERRUPT_STATUS_0_VHS_INT_STATUS_INTR                       (1)

// VVS pin Interrupt Status  This bit controls interrupt when VVS
//  rising/falling edge is detected.
//   0= Interrupt not detected
//   1= Interrupt detected
#define LWE24D_VI_INTERRUPT_STATUS_0_VVS_INT_STATUS                      8:8
#define LWE24D_VI_INTERRUPT_STATUS_0_VVS_INT_STATUS_NOINTR                     (0)
#define LWE24D_VI_INTERRUPT_STATUS_0_VVS_INT_STATUS_INTR                       (1)

// Vertical Counter Interrupt Status  (effective when VIDEO_SOURCE is HOST)
//  This bit controls interrupt when the
//  vertical counter threshold is reached.
#define LWE24D_VI_INTERRUPT_STATUS_0_V_COUNTER_INT_STATUS                        9:9
#define LWE24D_VI_INTERRUPT_STATUS_0_V_COUNTER_INT_STATUS_NOINTR                       (0)
#define LWE24D_VI_INTERRUPT_STATUS_0_V_COUNTER_INT_STATUS_INTR                 (1)

// Y-FIFO Threshold Interrupt Enable  This bit controls interrupt when the number
//  of filled locations in Y-FIFO is equal or
//  greater than the Y_FIFO_THRESHOLD value.
//   0= Interrupt not detected
//   1= Interrupt detected
#define LWE24D_VI_INTERRUPT_STATUS_0_Y_THRESHOLD_INT_STATUS                      10:10
#define LWE24D_VI_INTERRUPT_STATUS_0_Y_THRESHOLD_INT_STATUS_NOINTR                     (0)
#define LWE24D_VI_INTERRUPT_STATUS_0_Y_THRESHOLD_INT_STATUS_INTR                       (1)

// Buffer Done First Output Interrupt Status  This bit is set when a buffer has been
//  written to memory by the first output.
//   0= Interrupt not detected
//   1= Interrupt detected
#define LWE24D_VI_INTERRUPT_STATUS_0_BUFFER_FIRST_OUTPUT_INT_STATUS                      11:11
#define LWE24D_VI_INTERRUPT_STATUS_0_BUFFER_FIRST_OUTPUT_INT_STATUS_NOINTR                     (0)
#define LWE24D_VI_INTERRUPT_STATUS_0_BUFFER_FIRST_OUTPUT_INT_STATUS_INTR                       (1)

// Frame Done First Output Interrupt Status  This bit is set when a frame has been
//  written to memory by the first output.
//   0= Interrupt not detected
//   1= Interrupt detected
#define LWE24D_VI_INTERRUPT_STATUS_0_FRAME_FIRST_OUTPUT_INT_STATUS                       12:12
#define LWE24D_VI_INTERRUPT_STATUS_0_FRAME_FIRST_OUTPUT_INT_STATUS_NOINTR                      (0)
#define LWE24D_VI_INTERRUPT_STATUS_0_FRAME_FIRST_OUTPUT_INT_STATUS_INTR                        (1)

// Buffer Done Second Output Interrupt Status  This bit is set when a buffer has been
//  written to memory by the second output.
//   0= Interrupt not detected
//   1= Interrupt detected
#define LWE24D_VI_INTERRUPT_STATUS_0_BUFFER_SECOND_OUTPUT_INT_STATUS                     13:13
#define LWE24D_VI_INTERRUPT_STATUS_0_BUFFER_SECOND_OUTPUT_INT_STATUS_NOINTR                    (0)
#define LWE24D_VI_INTERRUPT_STATUS_0_BUFFER_SECOND_OUTPUT_INT_STATUS_INTR                      (1)

// Frame Done Second Output Interrupt Status  This bit is set when a frame has been
//  written to memory by the second output.
//   0= Interrupt not detected
//   1= Interrupt detected
#define LWE24D_VI_INTERRUPT_STATUS_0_FRAME_SECOND_OUTPUT_INT_STATUS                      14:14
#define LWE24D_VI_INTERRUPT_STATUS_0_FRAME_SECOND_OUTPUT_INT_STATUS_NOINTR                     (0)
#define LWE24D_VI_INTERRUPT_STATUS_0_FRAME_SECOND_OUTPUT_INT_STATUS_INTR                       (1)

// VI to EPP Error Interrupt Enable  This bit controls interrupt when the
//  VI drops data to the EPP because the
//  EPP is stalling the vi2epp bus and
//  data is coming from the pins
//   0= Interrupt not detected
//   1= Interrupt detected
#define LWE24D_VI_INTERRUPT_STATUS_0_EPP_ERROR_INT_STATUS                        15:15
#define LWE24D_VI_INTERRUPT_STATUS_0_EPP_ERROR_INT_STATUS_NOINTR                       (0)
#define LWE24D_VI_INTERRUPT_STATUS_0_EPP_ERROR_INT_STATUS_INTR                 (1)

// YUV420PA Error Interrupt Enable This bit shows the status of if the
//  VI does not average data because the
//  line buffer data is not ready from the
//  memory controller.  The VI will write
//  unaveraged data and will write the U,V
//  data from the even line in such cases.
//   0= Interrupt not detected
//   1= Interrupt detected
#define LWE24D_VI_INTERRUPT_STATUS_0_YUV420PA_ERROR_INT_STATUS                   16:16
#define LWE24D_VI_INTERRUPT_STATUS_0_YUV420PA_ERROR_INT_STATUS_NOINTR                  (0)
#define LWE24D_VI_INTERRUPT_STATUS_0_YUV420PA_ERROR_INT_STATUS_INTR                    (1)

// This bit shows the status of if the
//  VI dropped a buffer packet to the
//  peer communicating with the first memory
//  output
//   0= Interrupt not detected
//   1= Interrupt detected
#define LWE24D_VI_INTERRUPT_STATUS_0_FIRST_OUTPUT_PEER_STALL_INT_STATUS                  17:17
#define LWE24D_VI_INTERRUPT_STATUS_0_FIRST_OUTPUT_PEER_STALL_INT_STATUS_NOINTR                 (0)
#define LWE24D_VI_INTERRUPT_STATUS_0_FIRST_OUTPUT_PEER_STALL_INT_STATUS_INTR                   (1)

// This bit shows the status of if the
//  VI dropped a buffer packet to the
//  peer communicating with the second memory
//  output
//   0= Interrupt not detected
//   1= Interrupt detected
#define LWE24D_VI_INTERRUPT_STATUS_0_SECOND_OUTPUT_PEER_STALL_INT_STATUS                 18:18
#define LWE24D_VI_INTERRUPT_STATUS_0_SECOND_OUTPUT_PEER_STALL_INT_STATUS_NOINTR                        (0)
#define LWE24D_VI_INTERRUPT_STATUS_0_SECOND_OUTPUT_PEER_STALL_INT_STATUS_INTR                  (1)

// This bit shows the status of the condition when the
//  VI drops data to the Write Buffer DMA
//   0= Interrupt not detected
//   1= Interrupt detected
#define LWE24D_VI_INTERRUPT_STATUS_0_DMA_STALL_INT_STATUS                        19:19
#define LWE24D_VI_INTERRUPT_STATUS_0_DMA_STALL_INT_STATUS_NOINTR                       (0)
#define LWE24D_VI_INTERRUPT_STATUS_0_DMA_STALL_INT_STATUS_INTR                 (1)

// Top or Bottom Field Status  This bit specifies whether the last received
//  video data field is top field or bottom
//  field as defined by FIELD_TYPE bit. This bit
//  is forced to 0 if FIELD_DETECT is DISABLED
//  when VIDEO_SOURCE is VIP.
//  This bit cannot be reset by software by
//  writing a 1.
//   0= Bottom field received
//   1= Top field received
#define LWE24D_VI_INTERRUPT_STATUS_0_FIELD_STATUS                        20:20
#define LWE24D_VI_INTERRUPT_STATUS_0_FIELD_STATUS_BOTTOM                       (0)
#define LWE24D_VI_INTERRUPT_STATUS_0_FIELD_STATUS_TOP                  (1)

// This bit shows the status of the condition when the
//  Raise Stream 1 returns to the Host
//   0= Interrupt not detected
//   1= Interrupt detected
#define LWE24D_VI_INTERRUPT_STATUS_0_RAISE_STREAM_1_INT_STATUS                   21:21
#define LWE24D_VI_INTERRUPT_STATUS_0_RAISE_STREAM_1_INT_STATUS_NOINTR                  (0)
#define LWE24D_VI_INTERRUPT_STATUS_0_RAISE_STREAM_1_INT_STATUS_INTR                    (1)

// This bit shows the status of the condition when the
//  Raise Stream 2 returns to the Host
//   0= Interrupt not detected
//   1= Interrupt detected
#define LWE24D_VI_INTERRUPT_STATUS_0_RAISE_STREAM_2_INT_STATUS                   22:22
#define LWE24D_VI_INTERRUPT_STATUS_0_RAISE_STREAM_2_INT_STATUS_NOINTR                  (0)
#define LWE24D_VI_INTERRUPT_STATUS_0_RAISE_STREAM_2_INT_STATUS_INTR                    (1)

// This bit shows the status of the condition when the
//  ISDB-T vi input gets an upstream error (error from the tuner)
//   0= Interrupt not detected
//   1= Interrupt detected
#define LWE24D_VI_INTERRUPT_STATUS_0_TS_UPSTREAM_ERROR_INT_STATUS                        23:23
#define LWE24D_VI_INTERRUPT_STATUS_0_TS_UPSTREAM_ERROR_INT_STATUS_NOINTR                       (0)
#define LWE24D_VI_INTERRUPT_STATUS_0_TS_UPSTREAM_ERROR_INT_STATUS_INTR                 (1)

// This bit shows the status of the condition when the
//  ISDB-T input get an underrun error (START condition detected
//  prior to receiving a full packet)
//   0= Interrupt not detected
//   1= Interrupt detected
#define LWE24D_VI_INTERRUPT_STATUS_0_TS_UNDERRUN_ERROR_INT_STATUS                        24:24
#define LWE24D_VI_INTERRUPT_STATUS_0_TS_UNDERRUN_ERROR_INT_STATUS_NOINTR                       (0)
#define LWE24D_VI_INTERRUPT_STATUS_0_TS_UNDERRUN_ERROR_INT_STATUS_INTR                 (1)

// This bit shows the status of the condition when the
//  ISDB-T input get an overrun error (more bytes in packet than specified
//   0= Interrupt not detected
//   1= Interrupt detected
#define LWE24D_VI_INTERRUPT_STATUS_0_TS_OVERRUN_ERROR_INT_STATUS                 25:25
#define LWE24D_VI_INTERRUPT_STATUS_0_TS_OVERRUN_ERROR_INT_STATUS_NOINTR                        (0)
#define LWE24D_VI_INTERRUPT_STATUS_0_TS_OVERRUN_ERROR_INT_STATUS_INTR                  (1)

// This bit shows the status of the condition when the
//  ISDB-T input an other protocol error (ex:
//  total packet received is FEC_SIZE+BODY_SIZE but
//  the individual FEC portion != FEC_SIZE and
//  the individual BODY portion != BODY_SIZE
//   0= Interrupt not detected
//   1= Interrupt detected
#define LWE24D_VI_INTERRUPT_STATUS_0_TS_OTHER_PROTOCOL_ERROR_INT_STATUS                  26:26
#define LWE24D_VI_INTERRUPT_STATUS_0_TS_OTHER_PROTOCOL_ERROR_INT_STATUS_NOINTR                 (0)
#define LWE24D_VI_INTERRUPT_STATUS_0_TS_OTHER_PROTOCOL_ERROR_INT_STATUS_INTR                   (1)

// If FIRST_OUTPUT is dropping data to MC, INTR
//   will be set.
#define LWE24D_VI_INTERRUPT_STATUS_0_FIRST_OUTPUT_DROP_MC_DATA_INT_STATUS                        27:27
#define LWE24D_VI_INTERRUPT_STATUS_0_FIRST_OUTPUT_DROP_MC_DATA_INT_STATUS_NOINTR                       (0)
#define LWE24D_VI_INTERRUPT_STATUS_0_FIRST_OUTPUT_DROP_MC_DATA_INT_STATUS_INTR                 (1)

// If SECOND_OUTPUT is dropping data to MC, INTR
//   will be set.
#define LWE24D_VI_INTERRUPT_STATUS_0_SECOND_OUTPUT_DROP_MC_DATA_INT_STATUS                       28:28
#define LWE24D_VI_INTERRUPT_STATUS_0_SECOND_OUTPUT_DROP_MC_DATA_INT_STATUS_NOINTR                      (0)
#define LWE24D_VI_INTERRUPT_STATUS_0_SECOND_OUTPUT_DROP_MC_DATA_INT_STATUS_INTR                        (1)


#define LWE24D_VI_VIP_INPUT_STATUS_0                   (0x51)
// The number of lines received (hsyncs)
#define LWE24D_VI_VIP_INPUT_STATUS_0_LINE_COUNT                  15:0

// The number of frames received (vsyncs)
// Any write to this register, clears.
#define LWE24D_VI_VIP_INPUT_STATUS_0_FRAME_COUNT                 31:16


#define LWE24D_VI_VIDEO_BUFFER_STATUS_0                        (0x52)
// Buffer status
//  This specifies the buffer number of the
//  the last video data field written to memory
#define LWE24D_VI_VIDEO_BUFFER_STATUS_0_FIRST_VIDEO_BUFFER_STATUS                        7:0

// Buffer status
//  This specifies the buffer number of the
//  the last video data field written to memory
#define LWE24D_VI_VIDEO_BUFFER_STATUS_0_SECOND_VIDEO_BUFFER_STATUS                       15:8

// Write count of the Raw Stream Write FIFO
//  This is the fifo used to synchronize the
//  data coming from pads into the vi clock domain.
#define LWE24D_VI_VIDEO_BUFFER_STATUS_0_RAW_STREAM_WRITE_COUNT                   19:16

// This register controls VHS and VVS output when H/V syncs are generated internally in the
// VI module (VIDEO_SOURCE is VIP and SYNC_FORMAT is INTHVS).
// The generated VHS and VVS signal can be sent to external video source device and used
// to synchronize the video data transfer from the video source to the VI module. VHS and VVS
// pin should be configured in output mode to output the internally generated H/V syncs.
// Also in this case, the internally generate H/V syncs can be used by the VI module
// as horizontal and vertical reference signals for the incoming video data.

#define LWE24D_VI_SYNC_OUTPUT_0                        (0x53)
// This specifies VHS output pulse width in
//  term of number of VI clock cycles.
//  Programmed value is actual value - 1 so
//  valid value ranges from 1 to 8.
#define LWE24D_VI_SYNC_OUTPUT_0_VHS_OUTPUT_WIDTH                 2:0

// This specifies VHS output pulse period in
//  term of number of VI clock cycles.
//  Programmed value is actual value - 1 so
//  valid value ranges from 32 to 8192.
#define LWE24D_VI_SYNC_OUTPUT_0_VHS_OUTPUT_PERIOD                        15:3

// This specifies VVS output pulse width in
//  term of number of VHS cycles.
//  Programmed value is actual value - 1 so
//  valid value ranges from 1 to 8.
#define LWE24D_VI_SYNC_OUTPUT_0_VVS_OUTPUT_WIDTH                 18:16

// This specifies VVS output pulse period in
//  term of number of VHS cycles.
//  Programmed value is actual value - 1 so
//  valid value ranges from 2 to 4096.
#define LWE24D_VI_SYNC_OUTPUT_0_VVS_OUTPUT_PERIOD                        31:19


#define LWE24D_VI_VVS_OUTPUT_DELAY_0                   (0x54)
// This specifies the number of VI clock cycles
//  from leading edge of VHS to leading edge of
//  VVS.
//  Programmed value is actual value + 2 so
//  valid value ranges from -2 to 13.
#define LWE24D_VI_VVS_OUTPUT_DELAY_0_VVS_OUTPUT_DELAY                    3:0

// VI Pulse Width Modulation signal generation
// PWM signal generation logic can generate up to 128 pulses per line internally and the PWM
//  pulse select registers determines which of the 128 pulses will be output. Any of the 128
//  internally generated pulse can be independently selected as output if they occur within
//  one line time.
// PWM signal can be output on the VGP6 pin if VGP6 output is enabled and the output select
//  is set to PWM.
// The PWM will be triggered by the first vsync after the PWM_ENABLE bit has been set.

#define LWE24D_VI_PWM_CONTROL_0                        (0x55)
// PWM Enable  0= Disabled
//  1= Enabled
#define LWE24D_VI_PWM_CONTROL_0_PWM_ENABLE                       0:0
#define LWE24D_VI_PWM_CONTROL_0_PWM_ENABLE_DISABLED                    (0)
#define LWE24D_VI_PWM_CONTROL_0_PWM_ENABLE_ENABLED                     (1)

// PWM Direction  0= Incrementing
//  1= Decrementing
#define LWE24D_VI_PWM_CONTROL_0_PWM_DIRECTION                    4:4
#define LWE24D_VI_PWM_CONTROL_0_PWM_DIRECTION_INCR                     (0)
#define LWE24D_VI_PWM_CONTROL_0_PWM_DIRECTION_DECR                     (1)

// PWM High Pulse (1 to 16)
#define LWE24D_VI_PWM_CONTROL_0_PWM_HIGH_PULSE                   11:8

// PWM Low Pulse  (1 to 16)
// 19:16 reserved
#define LWE24D_VI_PWM_CONTROL_0_PWM_LOW_PULSE                    15:12

// PWM Mode Continous - after PWM is turned on, continue
//              through the PWM's 128 cycles
//              repeatedly until the pwm is turned off.
// Single - after PWM is turned on, cycle once through
//          the 128 cycles and stop.
// Counter - after PWM is turned on, cycle through
//           the 128 cycles PWM_COUNTER number of
//           times then stop.
// 23:22 reserved
#define LWE24D_VI_PWM_CONTROL_0_PWM_MODE                 21:20
#define LWE24D_VI_PWM_CONTROL_0_PWM_MODE_CONTINUOUS                    (0)
#define LWE24D_VI_PWM_CONTROL_0_PWM_MODE_SINGLE                        (1)
#define LWE24D_VI_PWM_CONTROL_0_PWM_MODE_COUNTER                       (2)

// PWM Counter
//  8-bit value used when PWM_MODE is set to COUNTER
//  to determine how many times the PWM will cycle
//  through the 128 cycles
//  before stopping.
#define LWE24D_VI_PWM_CONTROL_0_PWM_COUNTER                      31:24

// The next 4 registers select which of the internal 128 pulses to be output.
//  Each bit in the four registers correspond to one internal pulse.

#define LWE24D_VI_PWM_SELECT_PULSE_A_0                 (0x56)
// PWM Select bits 31 to 0
#define LWE24D_VI_PWM_SELECT_PULSE_A_0_PWM_SELECT_A                      31:0


#define LWE24D_VI_PWM_SELECT_PULSE_B_0                 (0x57)
// PWM Select bits 63 to 32
#define LWE24D_VI_PWM_SELECT_PULSE_B_0_PWM_SELECT_B                      31:0


#define LWE24D_VI_PWM_SELECT_PULSE_C_0                 (0x58)
// PWM Select bits 95 to 64
#define LWE24D_VI_PWM_SELECT_PULSE_C_0_PWM_SELECT_C                      31:0


#define LWE24D_VI_PWM_SELECT_PULSE_D_0                 (0x59)
// PWM Select bits 127 to 96
#define LWE24D_VI_PWM_SELECT_PULSE_D_0_PWM_SELECT_D                      31:0


#define LWE24D_VI_VI_DATA_INPUT_CONTROL_0                      (0x5a)
// Mask the VD[11:0] pin inputs to the VI core and ISP
// The mask is not applied to the Host GPIO read value
#define LWE24D_VI_VI_DATA_INPUT_CONTROL_0_VI_DATA_INPUT_MASK                     11:0


#define LWE24D_VI_PIN_INPUT_ENABLE_0                   (0x5b)
// VD0 pin Input Enable  This bit controls VD0 pin input.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD0_INPUT_ENABLE                    0:0
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD0_INPUT_ENABLE_DISABLED                 (0)
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD0_INPUT_ENABLE_ENABLED                  (1)

// VD1 pin Input Enable  This bit controls VD1 pin input.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD1_INPUT_ENABLE                    1:1
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD1_INPUT_ENABLE_DISABLED                 (0)
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD1_INPUT_ENABLE_ENABLED                  (1)

// VD2 pin Input Enable  This bit controls VD2 pin input.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD2_INPUT_ENABLE                    2:2
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD2_INPUT_ENABLE_DISABLED                 (0)
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD2_INPUT_ENABLE_ENABLED                  (1)

// VD3 pin Input Enable  This bit controls VD3 pin input.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD3_INPUT_ENABLE                    3:3
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD3_INPUT_ENABLE_DISABLED                 (0)
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD3_INPUT_ENABLE_ENABLED                  (1)

// VD4 pin Input Enable  This bit controls VD4 pin input.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD4_INPUT_ENABLE                    4:4
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD4_INPUT_ENABLE_DISABLED                 (0)
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD4_INPUT_ENABLE_ENABLED                  (1)

// VD5 pin Input Enable  This bit controls VD5 pin input.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD5_INPUT_ENABLE                    5:5
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD5_INPUT_ENABLE_DISABLED                 (0)
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD5_INPUT_ENABLE_ENABLED                  (1)

// VD6 pin Input Enable  This bit controls VD6 pin input.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD6_INPUT_ENABLE                    6:6
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD6_INPUT_ENABLE_DISABLED                 (0)
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD6_INPUT_ENABLE_ENABLED                  (1)

// VD7 pin Input Enable  This bit controls VD7 pin input.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD7_INPUT_ENABLE                    7:7
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD7_INPUT_ENABLE_DISABLED                 (0)
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD7_INPUT_ENABLE_ENABLED                  (1)

// VD8 pin Input Enable  This bit controls VD7 pin input.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD8_INPUT_ENABLE                    8:8
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD8_INPUT_ENABLE_DISABLED                 (0)
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD8_INPUT_ENABLE_ENABLED                  (1)

// VD9 pin Input Enable  This bit controls VD7 pin input.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD9_INPUT_ENABLE                    9:9
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD9_INPUT_ENABLE_DISABLED                 (0)
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD9_INPUT_ENABLE_ENABLED                  (1)

// VD10 pin Input Enable  This bit controls VD7 pin input.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD10_INPUT_ENABLE                   10:10
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD10_INPUT_ENABLE_DISABLED                        (0)
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD10_INPUT_ENABLE_ENABLED                 (1)

// VD11 pin Input Enable  This bit controls VD7 pin input.
//   0= Disabled
//   1= Enabled
// 12 reserved
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD11_INPUT_ENABLE                   11:11
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD11_INPUT_ENABLE_DISABLED                        (0)
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VD11_INPUT_ENABLE_ENABLED                 (1)

// VHS pin Input Enable  This bit controls VHS pin input.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VHS_INPUT_ENABLE                    13:13
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VHS_INPUT_ENABLE_DISABLED                 (0)
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VHS_INPUT_ENABLE_ENABLED                  (1)

// VVS pin Input Enable  This bit controls VVS pin input.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VVS_INPUT_ENABLE                    14:14
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VVS_INPUT_ENABLE_DISABLED                 (0)
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VVS_INPUT_ENABLE_ENABLED                  (1)

// VGP0 pin Input Enable  This bit controls VGP0 pin input.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VGP0_INPUT_ENABLE                   15:15
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VGP0_INPUT_ENABLE_DISABLED                        (0)
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VGP0_INPUT_ENABLE_ENABLED                 (1)

// VGP1 pin Input Enable  This bit controls VGP1 pin input.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VGP1_INPUT_ENABLE                   16:16
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VGP1_INPUT_ENABLE_DISABLED                        (0)
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VGP1_INPUT_ENABLE_ENABLED                 (1)

// VGP2 pin Input Enable  This bit controls VGP2 pin input.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VGP2_INPUT_ENABLE                   17:17
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VGP2_INPUT_ENABLE_DISABLED                        (0)
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VGP2_INPUT_ENABLE_ENABLED                 (1)

// VGP3 pin Input Enable  This bit controls VGP3 pin input.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VGP3_INPUT_ENABLE                   18:18
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VGP3_INPUT_ENABLE_DISABLED                        (0)
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VGP3_INPUT_ENABLE_ENABLED                 (1)

// VGP4 pin Input Enable  This bit controls VGP4 pin input.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VGP4_INPUT_ENABLE                   19:19
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VGP4_INPUT_ENABLE_DISABLED                        (0)
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VGP4_INPUT_ENABLE_ENABLED                 (1)

// VGP5 pin Input Enable  This bit controls VGP5 pin input.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VGP5_INPUT_ENABLE                   20:20
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VGP5_INPUT_ENABLE_DISABLED                        (0)
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VGP5_INPUT_ENABLE_ENABLED                 (1)

// VGP6 pin Input Enable  This bit controls VGP6 pin input.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VGP6_INPUT_ENABLE                   21:21
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VGP6_INPUT_ENABLE_DISABLED                        (0)
#define LWE24D_VI_PIN_INPUT_ENABLE_0_VGP6_INPUT_ENABLE_ENABLED                 (1)


#define LWE24D_VI_PIN_OUTPUT_ENABLE_0                  (0x5c)
// VD0 pin Output Enable  This bit controls VD0 pin output.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD0_OUTPUT_ENABLE                  0:0
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD0_OUTPUT_ENABLE_DISABLED                       (0)
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD0_OUTPUT_ENABLE_ENABLED                        (1)

// VD1 pin Output Enable  This bit controls VD1 pin output.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD1_OUTPUT_ENABLE                  1:1
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD1_OUTPUT_ENABLE_DISABLED                       (0)
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD1_OUTPUT_ENABLE_ENABLED                        (1)

// VD2 pin Output Enable  This bit controls VD2 pin output.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD2_OUTPUT_ENABLE                  2:2
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD2_OUTPUT_ENABLE_DISABLED                       (0)
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD2_OUTPUT_ENABLE_ENABLED                        (1)

// VD3 pin Output Enable  This bit controls VD3 pin output.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD3_OUTPUT_ENABLE                  3:3
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD3_OUTPUT_ENABLE_DISABLED                       (0)
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD3_OUTPUT_ENABLE_ENABLED                        (1)

// VD4 pin Output Enable  This bit controls VD4 pin output.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD4_OUTPUT_ENABLE                  4:4
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD4_OUTPUT_ENABLE_DISABLED                       (0)
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD4_OUTPUT_ENABLE_ENABLED                        (1)

// VD5 pin Output Enable  This bit controls VD5 pin output.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD5_OUTPUT_ENABLE                  5:5
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD5_OUTPUT_ENABLE_DISABLED                       (0)
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD5_OUTPUT_ENABLE_ENABLED                        (1)

// VD6 pin Output Enable  This bit controls VD6 pin output.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD6_OUTPUT_ENABLE                  6:6
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD6_OUTPUT_ENABLE_DISABLED                       (0)
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD6_OUTPUT_ENABLE_ENABLED                        (1)

// VD7 pin Output Enable  This bit controls VD7 pin output.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD7_OUTPUT_ENABLE                  7:7
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD7_OUTPUT_ENABLE_DISABLED                       (0)
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD7_OUTPUT_ENABLE_ENABLED                        (1)

// VD8 pin Output Enable  This bit controls VD7 pin output.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD8_OUTPUT_ENABLE                  8:8
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD8_OUTPUT_ENABLE_DISABLED                       (0)
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD8_OUTPUT_ENABLE_ENABLED                        (1)

// VD9 pin Output Enable  This bit controls VD7 pin output.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD9_OUTPUT_ENABLE                  9:9
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD9_OUTPUT_ENABLE_DISABLED                       (0)
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD9_OUTPUT_ENABLE_ENABLED                        (1)

// VD10 pin Output Enable  This bit controls VD7 pin output.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD10_OUTPUT_ENABLE                 10:10
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD10_OUTPUT_ENABLE_DISABLED                      (0)
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD10_OUTPUT_ENABLE_ENABLED                       (1)

// VD11 pin Output Enable  This bit controls VD7 pin output.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD11_OUTPUT_ENABLE                 11:11
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD11_OUTPUT_ENABLE_DISABLED                      (0)
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VD11_OUTPUT_ENABLE_ENABLED                       (1)

// VSCK pin Output Enable  This bit controls VSCK pin output.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VSCK_OUTPUT_ENABLE                 12:12
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VSCK_OUTPUT_ENABLE_DISABLED                      (0)
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VSCK_OUTPUT_ENABLE_ENABLED                       (1)

// VHS pin Output Enable  This bit controls VHS pin output.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VHS_OUTPUT_ENABLE                  13:13
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VHS_OUTPUT_ENABLE_DISABLED                       (0)
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VHS_OUTPUT_ENABLE_ENABLED                        (1)

// VVS pin Output Enable  This bit controls VVS pin output.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VVS_OUTPUT_ENABLE                  14:14
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VVS_OUTPUT_ENABLE_DISABLED                       (0)
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VVS_OUTPUT_ENABLE_ENABLED                        (1)

// VGP0 pin Output Enable  This bit controls VGP0 pin output.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VGP0_OUTPUT_ENABLE                 15:15
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VGP0_OUTPUT_ENABLE_DISABLED                      (0)
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VGP0_OUTPUT_ENABLE_ENABLED                       (1)

// VGP1 pin Output Enable  This bit controls VGP1 pin output.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VGP1_OUTPUT_ENABLE                 16:16
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VGP1_OUTPUT_ENABLE_DISABLED                      (0)
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VGP1_OUTPUT_ENABLE_ENABLED                       (1)

// VGP2 pin Output Enable  This bit controls VGP2 pin output.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VGP2_OUTPUT_ENABLE                 17:17
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VGP2_OUTPUT_ENABLE_DISABLED                      (0)
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VGP2_OUTPUT_ENABLE_ENABLED                       (1)

// VGP3 pin Output Enable  This bit controls VGP3 pin output.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VGP3_OUTPUT_ENABLE                 18:18
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VGP3_OUTPUT_ENABLE_DISABLED                      (0)
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VGP3_OUTPUT_ENABLE_ENABLED                       (1)

// VGP4 pin Output Enable  This bit controls VGP4 pin output.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VGP4_OUTPUT_ENABLE                 19:19
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VGP4_OUTPUT_ENABLE_DISABLED                      (0)
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VGP4_OUTPUT_ENABLE_ENABLED                       (1)

// VGP5 pin Output Enable  This bit controls VGP5 pin output.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VGP5_OUTPUT_ENABLE                 20:20
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VGP5_OUTPUT_ENABLE_DISABLED                      (0)
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VGP5_OUTPUT_ENABLE_ENABLED                       (1)

// VGP6 pin Output Enable  This bit controls VGP6 pin output.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VGP6_OUTPUT_ENABLE                 21:21
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VGP6_OUTPUT_ENABLE_DISABLED                      (0)
#define LWE24D_VI_PIN_OUTPUT_ENABLE_0_VGP6_OUTPUT_ENABLE_ENABLED                       (1)


//    0  reserved
#define LWE24D_VI_PIN_ILWERSION_0                      (0x5d)
// VHS pin Input Ilwersion   0= VHS input is not ilwerted
//      (VHS input is active high)
//   1= VHS input is ilwerted
//      (VHS input is active low)
#define LWE24D_VI_PIN_ILWERSION_0_VHS_IN_ILWERSION                       1:1
#define LWE24D_VI_PIN_ILWERSION_0_VHS_IN_ILWERSION_DISABLED                    (0)
#define LWE24D_VI_PIN_ILWERSION_0_VHS_IN_ILWERSION_ENABLED                     (1)

// VVS pin Input Ilwersion   0= VVS input is not ilwerted
//      (VVS input is active high)
//   1= VVS input is ilwerted
//      (VVS input is active low)
#define LWE24D_VI_PIN_ILWERSION_0_VVS_IN_ILWERSION                       2:2
#define LWE24D_VI_PIN_ILWERSION_0_VVS_IN_ILWERSION_DISABLED                    (0)
#define LWE24D_VI_PIN_ILWERSION_0_VVS_IN_ILWERSION_ENABLED                     (1)

// VSCK pin Output Ilwersion   0= VSCK output is not ilwerted
//   1= VSCK output is ilwerted
#define LWE24D_VI_PIN_ILWERSION_0_VSCK_OUT_ILWERSION                     16:16
#define LWE24D_VI_PIN_ILWERSION_0_VSCK_OUT_ILWERSION_DISABLED                  (0)
#define LWE24D_VI_PIN_ILWERSION_0_VSCK_OUT_ILWERSION_ENABLED                   (1)

// VHS pin Output Ilwersion   0= VHS output is not ilwerted
//      (VHS output is active high)
//   1= VHS output is ilwerted
//      (VHS output is active low)
#define LWE24D_VI_PIN_ILWERSION_0_VHS_OUT_ILWERSION                      17:17
#define LWE24D_VI_PIN_ILWERSION_0_VHS_OUT_ILWERSION_DISABLED                   (0)
#define LWE24D_VI_PIN_ILWERSION_0_VHS_OUT_ILWERSION_ENABLED                    (1)

// VVS pin Output Ilwersion   0= VVS output is not ilwerted
//      (VVS output is active high)
//   1= VVS output is ilwerted
//      (VVS output is active low)
#define LWE24D_VI_PIN_ILWERSION_0_VVS_OUT_ILWERSION                      18:18
#define LWE24D_VI_PIN_ILWERSION_0_VVS_OUT_ILWERSION_DISABLED                   (0)
#define LWE24D_VI_PIN_ILWERSION_0_VVS_OUT_ILWERSION_ENABLED                    (1)

// This register contains input data when the video camera interface pins are used as
// general-purpose input pins. The pin data read from this register is not affected by
// the pin input ilwersion bits.

#define LWE24D_VI_PIN_INPUT_DATA_0                     (0x5e)
// VD0 pin Input Data
//  (effective if VD0_INPUT_ENABLE is ENABLED)
//   0= VD0 input low
//   1= VD0 input high
#define LWE24D_VI_PIN_INPUT_DATA_0_VD0_INPUT_DATA                        0:0

// VD1 pin Input Data
//  (effective if VD1_INPUT_ENABLE is ENABLED)
//   0= VD1 input low
//   1= VD1 input high
#define LWE24D_VI_PIN_INPUT_DATA_0_VD1_INPUT_DATA                        1:1

// VD2 pin Input Data
//  (effective if VD2_INPUT_ENABLE is ENABLED)
//   0= VD2 input low
//   1= VD2 input high
#define LWE24D_VI_PIN_INPUT_DATA_0_VD2_INPUT_DATA                        2:2

// VD3 pin Input Data
//  (effective if VD3_INPUT_ENABLE is ENABLED)
//   0= VD3 input low
//   1= VD3 input high
#define LWE24D_VI_PIN_INPUT_DATA_0_VD3_INPUT_DATA                        3:3

// VD4 pin Input Data
//  (effective if VD4_INPUT_ENABLE is ENABLED)
//   0= VD4 input low
//   1= VD4 input high
#define LWE24D_VI_PIN_INPUT_DATA_0_VD4_INPUT_DATA                        4:4

// VD5 pin Input Data
//  (effective if VD5_INPUT_ENABLE is ENABLED)
//   0= VD5 input low
//   1= VD5 input high
#define LWE24D_VI_PIN_INPUT_DATA_0_VD5_INPUT_DATA                        5:5

// VD6 pin Input Data
//  (effective if VD6_INPUT_ENABLE is ENABLED)
//   0= VD6 input low
//   1= VD6 input high
#define LWE24D_VI_PIN_INPUT_DATA_0_VD6_INPUT_DATA                        6:6

// VD7 pin Input Data
//  (effective if VD7_INPUT_ENABLE is ENABLED)
//   0= VD7 input low
//   1= VD7 input high
#define LWE24D_VI_PIN_INPUT_DATA_0_VD7_INPUT_DATA                        7:7

// VD8 pin Input Data
//  (effective if VD8_INPUT_ENABLE is ENABLED)
//   0= VD8 input low
//   1= VD8 input high
#define LWE24D_VI_PIN_INPUT_DATA_0_VD8_INPUT_DATA                        8:8

// VD9 pin Input Data
//  (effective if VD9_INPUT_ENABLE is ENABLED)
//   0= VD9 input low
//   1= VD9 input high
#define LWE24D_VI_PIN_INPUT_DATA_0_VD9_INPUT_DATA                        9:9

// VD10 pin Input Data
//  (effective if VD10_INPUT_ENABLE is ENABLED)
//   0= VD10 input low
//   1= VD10 input high
#define LWE24D_VI_PIN_INPUT_DATA_0_VD10_INPUT_DATA                       10:10

// VD11 pin Input Data
//  (effective if VD11_INPUT_ENABLE is ENABLED)
//   0= VD11 input low
//   1= VD11 input high
#define LWE24D_VI_PIN_INPUT_DATA_0_VD11_INPUT_DATA                       11:11

// VSCK pin Input Data
//  (effective if VSCK_INPUT_ENABLE is ENABLED)
//   0= VSCK input low
//   1= VSCK input high
#define LWE24D_VI_PIN_INPUT_DATA_0_VSCK_INPUT_DATA                       12:12

// VHS pin Input Data
//  (effective if VHS_INPUT_ENABLE is ENABLED)
//   0= VHS input low
//   1= VHS input high
#define LWE24D_VI_PIN_INPUT_DATA_0_VHS_INPUT_DATA                        13:13

// VVS pin Input Data
//  (effective if VVS_INPUT_ENABLE is ENABLED)
//   0= VVS input low
//   1= VVS input high
#define LWE24D_VI_PIN_INPUT_DATA_0_VVS_INPUT_DATA                        14:14

// VGP0 pin Input Data
//  (effective if VGP0_INPUT_ENABLE is ENABLED)
//   0= VGP0 input low
//   1= VGP0 input high
#define LWE24D_VI_PIN_INPUT_DATA_0_VGP0_INPUT_DATA                       15:15

// VGP1 pin Input Data
//  (effective if VGP1_INPUT_ENABLE is ENABLED)
//   0= VGP1 input low
//   1= VGP1 input high
#define LWE24D_VI_PIN_INPUT_DATA_0_VGP1_INPUT_DATA                       16:16

// VGP2 pin Input Data
//  (effective if VGP2_INPUT_ENABLE is ENABLED)
//   0= VGP2 input low
//   1= VGP2 input high
#define LWE24D_VI_PIN_INPUT_DATA_0_VGP2_INPUT_DATA                       17:17

// VGP3 pin Input Data
//  (effective if VGP3_INPUT_ENABLE is ENABLED)
//   0= VGP3 input low
//   1= VGP3 input high
#define LWE24D_VI_PIN_INPUT_DATA_0_VGP3_INPUT_DATA                       18:18

// VGP4 pin Input Data
//  (effective if VGP4_INPUT_ENABLE is ENABLED)
//   0= VGP4 input low
//   1= VGP4 input high
#define LWE24D_VI_PIN_INPUT_DATA_0_VGP4_INPUT_DATA                       19:19

// VGP5 pin Input Data
//  (effective if VGP5_INPUT_ENABLE is ENABLED)
//   0= VGP5 input low
//   1= VGP5 input high
#define LWE24D_VI_PIN_INPUT_DATA_0_VGP5_INPUT_DATA                       20:20

// VGP6 pin Input Data
//  (effective if VGP6_INPUT_ENABLE is ENABLED)
//   0= VGP6 input low
//   1= VGP6 input high
#define LWE24D_VI_PIN_INPUT_DATA_0_VGP6_INPUT_DATA                       21:21

// This register contains output data when the video camera interface pins are used as
// general-purpose output pins. When a bit in this register is written, the data bits can be
// output on the corresponding pin if the corresponding pin output buffer is enabled and the
// pin output control select bits are programmed to output the bit in this register.
// The output signal at the pin IS affected by the corresponding pin output ilwersion bit.

#define LWE24D_VI_PIN_OUTPUT_DATA_0                    (0x5f)
// VD0 pin Output Data
//  (effective if VD0_OUTPUT_ENABLE is ENABLED
//   and VD0_OUTPUT_SELECT is DATA)
#define LWE24D_VI_PIN_OUTPUT_DATA_0_VD0_OUTPUT_DATA                      0:0

// VD1 pin Output Data
//  (effective if VD1_OUTPUT_ENABLE is ENABLED
//   and VD1_OUTPUT_SELECT is DATA)
#define LWE24D_VI_PIN_OUTPUT_DATA_0_VD1_OUTPUT_DATA                      1:1

// VD2 pin Output Data
//  (effective if VD2_OUTPUT_ENABLE is ENABLED
//   and VD2_OUTPUT_SELECT is DATA)
#define LWE24D_VI_PIN_OUTPUT_DATA_0_VD2_OUTPUT_DATA                      2:2

// VD3 pin Output Data
//  (effective if VD3_OUTPUT_ENABLE is ENABLED
//   and VD3_OUTPUT_SELECT is DATA)
#define LWE24D_VI_PIN_OUTPUT_DATA_0_VD3_OUTPUT_DATA                      3:3

// VD4 pin Output Data
//  (effective if VD4_OUTPUT_ENABLE is ENABLED
//   and VD4_OUTPUT_SELECT is DATA)
#define LWE24D_VI_PIN_OUTPUT_DATA_0_VD4_OUTPUT_DATA                      4:4

// VD5 pin Output Data
//  (effective if VD5_OUTPUT_ENABLE is ENABLED
//   and VD5_OUTPUT_SELECT is DATA)
#define LWE24D_VI_PIN_OUTPUT_DATA_0_VD5_OUTPUT_DATA                      5:5

// VD6 pin Output Data
//  (effective if VD6_OUTPUT_ENABLE is ENABLED
//   and VD6_OUTPUT_SELECT is DATA)
#define LWE24D_VI_PIN_OUTPUT_DATA_0_VD6_OUTPUT_DATA                      6:6

// VD7 pin Output Data
//  (effective if VD7_OUTPUT_ENABLE is ENABLED
//   and VD7_OUTPUT_SELECT is DATA)
#define LWE24D_VI_PIN_OUTPUT_DATA_0_VD7_OUTPUT_DATA                      7:7

// VD8 pin Output Data
//  (effective if VD8_OUTPUT_ENABLE is ENABLED
//   and VD8_OUTPUT_SELECT is DATA)
#define LWE24D_VI_PIN_OUTPUT_DATA_0_VD8_OUTPUT_DATA                      8:8

// VD9 pin Output Data
//  (effective if VD9_OUTPUT_ENABLE is ENABLED
//   and VD9_OUTPUT_SELECT is DATA)
#define LWE24D_VI_PIN_OUTPUT_DATA_0_VD9_OUTPUT_DATA                      9:9

// VD10 pin Output Data
//  (effective if VD10_OUTPUT_ENABLE is ENABLED
//   and VD10_OUTPUT_SELECT is DATA)
#define LWE24D_VI_PIN_OUTPUT_DATA_0_VD10_OUTPUT_DATA                     10:10

// VD11 pin Output Data
//  (effective if VD11_OUTPUT_ENABLE is ENABLED
//   and VD11_OUTPUT_SELECT is DATA)
#define LWE24D_VI_PIN_OUTPUT_DATA_0_VD11_OUTPUT_DATA                     11:11

// VSCK pin Output Data
//  (effective if VSCK_OUTPUT_ENABLE is ENABLED
//   and VSCK_OUTPUT_SELECT is DATA)
#define LWE24D_VI_PIN_OUTPUT_DATA_0_VSCK_OUTPUT_DATA                     12:12

// VHS pin Output Data
//  (effective if VHS_OUTPUT_ENABLE is ENABLED
//   and VHS_OUTPUT_SELECT is DATA)
#define LWE24D_VI_PIN_OUTPUT_DATA_0_VHS_OUTPUT_DATA                      13:13

// VVS pin Output Data
//  (effective if VVS_OUTPUT_ENABLE is ENABLED
//   and VVS_OUTPUT_SELECT is DATA)
#define LWE24D_VI_PIN_OUTPUT_DATA_0_VVS_OUTPUT_DATA                      14:14

// VGP0 pin Output Data
//  (effective if VGP0_OUTPUT_ENABLE is ENABLED
//   and VGP0_OUTPUT_SELECT is DATA)
#define LWE24D_VI_PIN_OUTPUT_DATA_0_VGP0_OUTPUT_DATA                     15:15

// VGP1 pin Output Data
//  (effective if VGP1_OUTPUT_ENABLE is ENABLED
//   and VGP1_OUTPUT_SELECT is DATA)
#define LWE24D_VI_PIN_OUTPUT_DATA_0_VGP1_OUTPUT_DATA                     16:16

// VGP2 pin Output Data
//  (effective if VGP2_OUTPUT_ENABLE is ENABLED
//   and VGP2_OUTPUT_SELECT is DATA)
#define LWE24D_VI_PIN_OUTPUT_DATA_0_VGP2_OUTPUT_DATA                     17:17

// VGP3 pin Output Data
//  (effective if VGP3_OUTPUT_ENABLE is ENABLED
//   and VGP3_OUTPUT_SELECT is DATA)
#define LWE24D_VI_PIN_OUTPUT_DATA_0_VGP3_OUTPUT_DATA                     18:18

// VGP4 pin Output Data
//  (effective if VGP4_OUTPUT_ENABLE is ENABLED
//   and VGP4_OUTPUT_SELECT is DATA)
#define LWE24D_VI_PIN_OUTPUT_DATA_0_VGP4_OUTPUT_DATA                     19:19

// VGP5 pin Output Data
//  (effective if VGP5_OUTPUT_ENABLE is ENABLED
//   and VGP5_OUTPUT_SELECT is DATA)
#define LWE24D_VI_PIN_OUTPUT_DATA_0_VGP5_OUTPUT_DATA                     20:20

// VGP6 pin Output Data
//  (effective if VGP6_OUTPUT_ENABLE is ENABLED
//   and VGP6_OUTPUT_SELECT is DATA)
#define LWE24D_VI_PIN_OUTPUT_DATA_0_VGP6_OUTPUT_DATA                     21:21


// This is the mux select used at the Pad Macro
// For VCLK, VHSYNC, VVSYNC
// Selects between the register programmed GPIO outputs (set to 0)
// and the internally generated viclk, hsync, vsync (set to 1)
// For VGP1-VGP2
// Selects between the I^2C outputs (set to 0)
// and the VI register programmed GPIO outputs (set to 1)
// For VD0-VD11
// Reserved for future use
// data pins output will be driven by GPIO outputs if enabled
#define LWE24D_VI_PIN_OUTPUT_SELECT_0                  (0x60)
// Pin Output Select VD0
#define LWE24D_VI_PIN_OUTPUT_SELECT_0_PIN_OUTPUT_SELECT_vd0                      0:0

// Pin Output Select VD1
#define LWE24D_VI_PIN_OUTPUT_SELECT_0_PIN_OUTPUT_SELECT_vd1                      1:1

// Pin Output Select VD2
#define LWE24D_VI_PIN_OUTPUT_SELECT_0_PIN_OUTPUT_SELECT_vd2                      2:2

// Pin Output Select VD3
#define LWE24D_VI_PIN_OUTPUT_SELECT_0_PIN_OUTPUT_SELECT_vd3                      3:3

// Pin Output Select VD4
#define LWE24D_VI_PIN_OUTPUT_SELECT_0_PIN_OUTPUT_SELECT_vd4                      4:4

// Pin Output Select VD5
#define LWE24D_VI_PIN_OUTPUT_SELECT_0_PIN_OUTPUT_SELECT_vd5                      5:5

// Pin Output Select VD6
#define LWE24D_VI_PIN_OUTPUT_SELECT_0_PIN_OUTPUT_SELECT_vd6                      6:6

// Pin Output Select VD7
#define LWE24D_VI_PIN_OUTPUT_SELECT_0_PIN_OUTPUT_SELECT_vd7                      7:7

// Pin Output Select VD8
#define LWE24D_VI_PIN_OUTPUT_SELECT_0_PIN_OUTPUT_SELECT_vd8                      8:8

// Pin Output Select VD9
#define LWE24D_VI_PIN_OUTPUT_SELECT_0_PIN_OUTPUT_SELECT_vd9                      9:9

// Pin Output Select VD10
#define LWE24D_VI_PIN_OUTPUT_SELECT_0_PIN_OUTPUT_SELECT_vd10                     10:10

// Pin Output Select VD11
#define LWE24D_VI_PIN_OUTPUT_SELECT_0_PIN_OUTPUT_SELECT_vd11                     11:11

// Pin Output Select VCLK
#define LWE24D_VI_PIN_OUTPUT_SELECT_0_PIN_OUTPUT_SELECT_vclk                     12:12

// Pin Output Select VHSYNC
#define LWE24D_VI_PIN_OUTPUT_SELECT_0_PIN_OUTPUT_SELECT_vhs                      13:13

// Pin Output Select VVSYNC
#define LWE24D_VI_PIN_OUTPUT_SELECT_0_PIN_OUTPUT_SELECT_vvs                      14:14

// Pin Output Select VGP0
//  0 = VGP0 output register
//  1 = refclk
#define LWE24D_VI_PIN_OUTPUT_SELECT_0_PIN_OUTPUT_SELECT_vgp0                     15:15

// Pin Output Select VGP1
//  0 = I^2C SCK pin
//  1 = 1'b0
#define LWE24D_VI_PIN_OUTPUT_SELECT_0_PIN_OUTPUT_SELECT_vgp1                     16:16

// Pin Output Select VGP2
//  0 = I^2C SDA pin
//  1 = 1'b0
#define LWE24D_VI_PIN_OUTPUT_SELECT_0_PIN_OUTPUT_SELECT_vgp2                     17:17

// Pin Output Select VGP3
//  0 = VGP3 output register
//  1 = 1'b0
#define LWE24D_VI_PIN_OUTPUT_SELECT_0_PIN_OUTPUT_SELECT_vgp3                     18:18

// Pin Output Select VGP4
//  0 = VGP4 output register
//  1 = 1'b0
#define LWE24D_VI_PIN_OUTPUT_SELECT_0_PIN_OUTPUT_SELECT_vgp4                     19:19

// Pin Output Select VGP5
//  0 = VGP5 output register
//  1 = 1'b0
#define LWE24D_VI_PIN_OUTPUT_SELECT_0_PIN_OUTPUT_SELECT_vgp5                     20:20

// Pin Output Select VGP6   0= select VGP6 register data out
//   1= select PWM out
#define LWE24D_VI_PIN_OUTPUT_SELECT_0_PIN_OUTPUT_SELECT_vgp6                     21:21
#define LWE24D_VI_PIN_OUTPUT_SELECT_0_PIN_OUTPUT_SELECT_vgp6_DATA                      (0)
#define LWE24D_VI_PIN_OUTPUT_SELECT_0_PIN_OUTPUT_SELECT_vgp6_PWM                       (1)

// raise vectors are received from host. If host is the input source, host will send
// a raise vector at the end of a line, and VI return it when that has been written to memory.
// A raise written when decimation or averaging is selected in vi, is not supported.
// If Video Input Port is the input source, host should program raise vectors to either raise
// at buffer end or at frame end.
// Since there are 2 memory outputs for vi, there are two separate raise vectors for buffer/frame.

#define LWE24D_VI_RAISE_VIP_BUFFER_FIRST_OUTPUT_0                      (0x61)
#define LWE24D_VI_RAISE_VIP_BUFFER_FIRST_OUTPUT_0_RAISE_BUFFER_1_VECTOR                  4:0

#define LWE24D_VI_RAISE_VIP_BUFFER_FIRST_OUTPUT_0_RAISE_BUFFER_1_CHANNEL                 19:16


#define LWE24D_VI_RAISE_VIP_FRAME_FIRST_OUTPUT_0                       (0x62)
#define LWE24D_VI_RAISE_VIP_FRAME_FIRST_OUTPUT_0_RAISE_FRAME_1_VECTOR                    4:0

#define LWE24D_VI_RAISE_VIP_FRAME_FIRST_OUTPUT_0_RAISE_FRAME_1_CHANNEL                   19:16


#define LWE24D_VI_RAISE_VIP_BUFFER_SECOND_OUTPUT_0                     (0x63)
#define LWE24D_VI_RAISE_VIP_BUFFER_SECOND_OUTPUT_0_RAISE_BUFFER_2_VECTOR                 4:0

#define LWE24D_VI_RAISE_VIP_BUFFER_SECOND_OUTPUT_0_RAISE_BUFFER_2_CHANNEL                        19:16


#define LWE24D_VI_RAISE_VIP_FRAME_SECOND_OUTPUT_0                      (0x64)
#define LWE24D_VI_RAISE_VIP_FRAME_SECOND_OUTPUT_0_RAISE_FRAME_2_VECTOR                   4:0

#define LWE24D_VI_RAISE_VIP_FRAME_SECOND_OUTPUT_0_RAISE_FRAME_2_CHANNEL                  19:16


#define LWE24D_VI_RAISE_HOST_FIRST_OUTPUT_0                    (0x65)
#define LWE24D_VI_RAISE_HOST_FIRST_OUTPUT_0_RAISE_HOST_1_VECTOR                  4:0

#define LWE24D_VI_RAISE_HOST_FIRST_OUTPUT_0_RAISE_HOST_1_CHANNEL                 19:16


#define LWE24D_VI_RAISE_HOST_SECOND_OUTPUT_0                   (0x66)
#define LWE24D_VI_RAISE_HOST_SECOND_OUTPUT_0_RAISE_HOST_2_VECTOR                 4:0

#define LWE24D_VI_RAISE_HOST_SECOND_OUTPUT_0_RAISE_HOST_2_CHANNEL                        19:16

// EPP receives the raise request via the Simple Stream Video Data bus
// see arepp.spec for details
// This raise needs to be written during the horizontal blanking period. (After end of line.)
// This is only valid if the input source is host.

#define LWE24D_VI_RAISE_EPP_0                  (0x67)
#define LWE24D_VI_RAISE_EPP_0_RAISE_EPP_VECTOR                   4:0

#define LWE24D_VI_RAISE_EPP_0_RAISE_EPP_CHANNEL                  19:16

// For Parallel VIP input, limitation for vsync and hsync has to be followed to avoid ISP hang for AP15:
// SW must always program parallel cameras (including the VIP pattern generator) in a way that
// avoids simultaneous hsync and vsync active edges. copied from bug:361730

#define LWE24D_VI_CAMERA_CONTROL_0                     (0x68)
// VI camera input module Enable   0= Ignored - use the STOP_CAPTURE to turn off the capturing
//   1= Enabled
// Write a 1'b1 to this register to enable
// the camera interface to start capturing data.
#define LWE24D_VI_CAMERA_CONTROL_0_VIP_ENABLE                    0:0
#define LWE24D_VI_CAMERA_CONTROL_0_VIP_ENABLE_DISABLED                 (0)
#define LWE24D_VI_CAMERA_CONTROL_0_VIP_ENABLE_ENABLED                  (1)

// Test Mode Enable   0= Disabled
//   1= Enabled
#define LWE24D_VI_CAMERA_CONTROL_0_TEST_MODE_ENABLE                      1:1
#define LWE24D_VI_CAMERA_CONTROL_0_TEST_MODE_ENABLE_DISABLED                   (0)
#define LWE24D_VI_CAMERA_CONTROL_0_TEST_MODE_ENABLE_ENABLED                    (1)

// Disables camera capturing VI_ENABLE after the  next end of frame.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_CAMERA_CONTROL_0_STOP_CAPTURE                  2:2
#define LWE24D_VI_CAMERA_CONTROL_0_STOP_CAPTURE_DISABLED                       (0)
#define LWE24D_VI_CAMERA_CONTROL_0_STOP_CAPTURE_ENABLED                        (1)

// **** Enable bit should be to host together with other module enables in LW flow.
// **** Test mode is not needed in LW flow but the enable bit can be replaced with debug bus
//      enable.

#define LWE24D_VI_VI_ENABLE_0                  (0x69)
// First Output to Memory   0= Enabled
//   1= Disabled
#define LWE24D_VI_VI_ENABLE_0_FIRST_OUTPUT_TO_MEMORY                     0:0
#define LWE24D_VI_VI_ENABLE_0_FIRST_OUTPUT_TO_MEMORY_ENABLED                   (0)
#define LWE24D_VI_VI_ENABLE_0_FIRST_OUTPUT_TO_MEMORY_DISABLED                  (1)

// SW enable flow control for output1
#define LWE24D_VI_VI_ENABLE_0_SW_FLOW_CONTROL_OUT1                       1:1
#define LWE24D_VI_VI_ENABLE_0_SW_FLOW_CONTROL_OUT1_DISABLE                     (0)
#define LWE24D_VI_VI_ENABLE_0_SW_FLOW_CONTROL_OUT1_ENABLE                      (1)


#define LWE24D_VI_VI_ENABLE_2_0                        (0x6a)
// Second Output to Memory   0= Enabled
//   1= Disabled
//  Disabling output to memory may be set
//  if only output to encoder pre-processor
//  is needed. This will also power-down
//  all logic which is only used to send
//  output data to memory.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_VI_ENABLE_2_0_SECOND_OUTPUT_TO_MEMORY                  0:0
#define LWE24D_VI_VI_ENABLE_2_0_SECOND_OUTPUT_TO_MEMORY_ENABLED                        (0)
#define LWE24D_VI_VI_ENABLE_2_0_SECOND_OUTPUT_TO_MEMORY_DISABLED                       (1)

// SW enable flow control for output2
#define LWE24D_VI_VI_ENABLE_2_0_SW_FLOW_CONTROL_OUT2                     1:1
#define LWE24D_VI_VI_ENABLE_2_0_SW_FLOW_CONTROL_OUT2_DISABLE                   (0)
#define LWE24D_VI_VI_ENABLE_2_0_SW_FLOW_CONTROL_OUT2_ENABLE                    (1)


#define LWE24D_VI_VI_RAISE_0                   (0x6b)
// Makes Raises edge triggered not level sensitive  i.e. only return raise at the end of frame, not
//  in the middle of the v-blank time.
//   0= Disabled
//   1= Enabled
#define LWE24D_VI_VI_RAISE_0_RAISE_ON_EDGE                       0:0
#define LWE24D_VI_VI_RAISE_0_RAISE_ON_EDGE_DISABLED                    (0)
#define LWE24D_VI_VI_RAISE_0_RAISE_ON_EDGE_ENABLED                     (1)

// **** Host YUV FIFO offsets.  This register space is used for Host Video Data writes.
// **** YUV 4:2:0 planar for re-encoding as well as YUV 4:2:2 data

#define LWE24D_VI_Y_FIFO_WRITE_0                       (0x6c)
#define LWE24D_VI_Y_FIFO_WRITE_0_Y_FIFO_DATA                     31:0


#define LWE24D_VI_U_FIFO_WRITE_0                       (0x6d)
#define LWE24D_VI_U_FIFO_WRITE_0_U_FIFO_DATA                     31:0


#define LWE24D_VI_V_FIFO_WRITE_0                       (0x6e)
#define LWE24D_VI_V_FIFO_WRITE_0_V_FIFO_DATA                     31:0

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

#define LWE24D_VI_VI_MCCIF_FIFOCTRL_0                  (0x6f)
#define LWE24D_VI_VI_MCCIF_FIFOCTRL_0_VI_MCCIF_WRCL_MCLE2X                       0:0
#define LWE24D_VI_VI_MCCIF_FIFOCTRL_0_VI_MCCIF_WRCL_MCLE2X_DISABLE                     (0)
#define LWE24D_VI_VI_MCCIF_FIFOCTRL_0_VI_MCCIF_WRCL_MCLE2X_ENABLE                      (1)

#define LWE24D_VI_VI_MCCIF_FIFOCTRL_0_VI_MCCIF_RDMC_RDFAST                       1:1
#define LWE24D_VI_VI_MCCIF_FIFOCTRL_0_VI_MCCIF_RDMC_RDFAST_DISABLE                     (0)
#define LWE24D_VI_VI_MCCIF_FIFOCTRL_0_VI_MCCIF_RDMC_RDFAST_ENABLE                      (1)

#define LWE24D_VI_VI_MCCIF_FIFOCTRL_0_VI_MCCIF_WRMC_CLLE2X                       2:2
#define LWE24D_VI_VI_MCCIF_FIFOCTRL_0_VI_MCCIF_WRMC_CLLE2X_DISABLE                     (0)
#define LWE24D_VI_VI_MCCIF_FIFOCTRL_0_VI_MCCIF_WRMC_CLLE2X_ENABLE                      (1)

#define LWE24D_VI_VI_MCCIF_FIFOCTRL_0_VI_MCCIF_RDCL_RDFAST                       3:3
#define LWE24D_VI_VI_MCCIF_FIFOCTRL_0_VI_MCCIF_RDCL_RDFAST_DISABLE                     (0)
#define LWE24D_VI_VI_MCCIF_FIFOCTRL_0_VI_MCCIF_RDCL_RDFAST_ENABLE                      (1)

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

#define LWE24D_VI_TIMEOUT_WCOAL_VI_0                   (0x70)
#define LWE24D_VI_TIMEOUT_WCOAL_VI_0_VIWSB_WCOAL_TMVAL                   7:0

#define LWE24D_VI_TIMEOUT_WCOAL_VI_0_VIWU_WCOAL_TMVAL                    15:8

#define LWE24D_VI_TIMEOUT_WCOAL_VI_0_VIWV_WCOAL_TMVAL                    23:16

#define LWE24D_VI_TIMEOUT_WCOAL_VI_0_VIWY_WCOAL_TMVAL                    31:24

// Memory Client High-Priority Control Register
// This register exists only for clients with high-priority. Reset values are 0 (disabled).
// The high-priority should be enabled for hard real-time clients only. The values to program
// depend on the client bandwidth requirement and the client versus memory controllers clolck ratio.
// The high-priority is set if the number of entries in the return data fifo is under the threshold.
// The high-priority assertion can be delayed by a number of memory clock cycles indicated by the timer.
// This creates an hysteresis effect, avoiding setting the high-priority for very short periods of time,
// which may or may not be desirable.

#define LWE24D_VI_MCCIF_VIRUV_HP_0                     (0x71)
#define LWE24D_VI_MCCIF_VIRUV_HP_0_CBR_VIRUV2MC_HPTH                     3:0

#define LWE24D_VI_MCCIF_VIRUV_HP_0_CBR_VIRUV2MC_HPTM                     21:16

// Memory Client High-Priority Control Register
// This register exists only for clients with high-priority. Reset values are 0 (disabled).
// The high-priority should be enabled for hard real-time clients only. The values to program
// depend on the client bandwidth requirement and the client versus memory controllers clolck ratio.
// The high-priority is set if the number of entries in the data fifo is higher than the threshold.

#define LWE24D_VI_MCCIF_VIWSB_HP_0                     (0x72)
#define LWE24D_VI_MCCIF_VIWSB_HP_0_CBW_VIWSB2MC_HPTH                     6:0

// Memory Client High-Priority Control Register
// This register exists only for clients with high-priority. Reset values are 0 (disabled).
// The high-priority should be enabled for hard real-time clients only. The values to program
// depend on the client bandwidth requirement and the client versus memory controllers clolck ratio.
// The high-priority is set if the number of entries in the data fifo is higher than the threshold.

#define LWE24D_VI_MCCIF_VIWU_HP_0                      (0x73)
#define LWE24D_VI_MCCIF_VIWU_HP_0_CBW_VIWU2MC_HPTH                       6:0

// Memory Client High-Priority Control Register
// This register exists only for clients with high-priority. Reset values are 0 (disabled).
// The high-priority should be enabled for hard real-time clients only. The values to program
// depend on the client bandwidth requirement and the client versus memory controllers clolck ratio.
// The high-priority is set if the number of entries in the data fifo is higher than the threshold.

#define LWE24D_VI_MCCIF_VIWV_HP_0                      (0x74)
#define LWE24D_VI_MCCIF_VIWV_HP_0_CBW_VIWV2MC_HPTH                       6:0

// Memory Client High-Priority Control Register
// This register exists only for clients with high-priority. Reset values are 0 (disabled).
// The high-priority should be enabled for hard real-time clients only. The values to program
// depend on the client bandwidth requirement and the client versus memory controllers clolck ratio.
// The high-priority is set if the number of entries in the data fifo is higher than the threshold.

#define LWE24D_VI_MCCIF_VIWY_HP_0                      (0x75)
#define LWE24D_VI_MCCIF_VIWY_HP_0_CBW_VIWY2MC_HPTH                       6:0

// CSI Raise vectors

#define LWE24D_VI_CSI_PPA_RAISE_FRAME_START_0                  (0x76)
//   Raise returned by VI when CSI PPA
//   issues a frame start to consumer.
#define LWE24D_VI_CSI_PPA_RAISE_FRAME_START_0_CSI_PPA_RAISE_FRAME_START_VECTOR                   4:0

//   Number of frame start since last raise >= count for raise to be returned
#define LWE24D_VI_CSI_PPA_RAISE_FRAME_START_0_CSI_PPA_RAISE_FRAME_START_COUNT                    15:8

//   Raise return channel
#define LWE24D_VI_CSI_PPA_RAISE_FRAME_START_0_CSI_PPA_RAISE_FRAME_START_CHANNEL                  19:16


#define LWE24D_VI_CSI_PPA_RAISE_FRAME_END_0                    (0x77)
//   Raise returned by VI when CSI PPA
//   issues a frame end to consumer.
#define LWE24D_VI_CSI_PPA_RAISE_FRAME_END_0_CSI_PPA_RAISE_FRAME_END_VECTOR                       4:0

//   Number of frame end since last raise >= count for raise to be returned
#define LWE24D_VI_CSI_PPA_RAISE_FRAME_END_0_CSI_PPA_RAISE_FRAME_END_COUNT                        15:8

//   Raise return channel
#define LWE24D_VI_CSI_PPA_RAISE_FRAME_END_0_CSI_PPA_RAISE_FRAME_END_CHANNEL                      19:16


#define LWE24D_VI_CSI_PPB_RAISE_FRAME_START_0                  (0x78)
//   Raise returned by VI when CSI PPB
//   issues a frame start to consumer.
#define LWE24D_VI_CSI_PPB_RAISE_FRAME_START_0_CSI_PPB_RAISE_FRAME_START_VECTOR                   4:0

//   Number of frame start since last raise >= count for raise to be returned
#define LWE24D_VI_CSI_PPB_RAISE_FRAME_START_0_CSI_PPB_RAISE_FRAME_START_COUNT                    15:8

//   Raise return channel
#define LWE24D_VI_CSI_PPB_RAISE_FRAME_START_0_CSI_PPB_RAISE_FRAME_START_CHANNEL                  19:16


#define LWE24D_VI_CSI_PPB_RAISE_FRAME_END_0                    (0x79)
//   Raise returned by VI when CSI PPB
//   issues a frame end to consumer.
#define LWE24D_VI_CSI_PPB_RAISE_FRAME_END_0_CSI_PPB_RAISE_FRAME_END_VECTOR                       4:0

//   Number of frame end since last raise >= count for raise to be returned
#define LWE24D_VI_CSI_PPB_RAISE_FRAME_END_0_CSI_PPB_RAISE_FRAME_END_COUNT                        15:8

//   Raise return channel
#define LWE24D_VI_CSI_PPB_RAISE_FRAME_END_0_CSI_PPB_RAISE_FRAME_END_CHANNEL                      19:16

// This register defines the horizontal captured (active) area of the input video source with respect to
//  horizontal sync. (This is for CSI data.)

#define LWE24D_VI_CSI_PPA_H_ACTIVE_0                   (0x7a)
// Horizontal Active Start (offset to active)
//  This parameter specifies the number of
//  clock active edges from horizontal
//  sync active edge to the first horizontal
//  active pixel. If programmed to 0, the
//  first active line starts after the first
//  active clock edge following the horizontal
//  sync active edge.
#define LWE24D_VI_CSI_PPA_H_ACTIVE_0_CSI_PPA_H_ACTIVE_START                      12:0

// Horizontal Active Period
//  This parameter specifies the number of
//  pixels in the horizontal active area.
//  H_ACTIVE_START + H_ACTIVE_PERIOD should be
//  less than 2^LW_VI_H_IN (or 8192). This parameter
//  should be programmed with an even number
//  (bit 16 is ignored internally).
#define LWE24D_VI_CSI_PPA_H_ACTIVE_0_CSI_PPA_H_ACTIVE_PERIOD                     28:16

// This register defines the vertical captured (active) area of the input video source with respect to
//  vertical sync. (This is for CSI data.)

#define LWE24D_VI_CSI_PPA_V_ACTIVE_0                   (0x7b)
// Vertical Active Start (offset to active)
//  This parameter specifies the number of
//  horizontal sync active edges from vertical
//  sync active edge to the first vertical
//  active line. If programmed to 0, the
//  first active line starts after the first
//  horizontal sync active edge following
//  the vertical sync active edge.
#define LWE24D_VI_CSI_PPA_V_ACTIVE_0_CSI_PPA_V_ACTIVE_START                      12:0

// Vertical Active Period
//  This parameter specifies the number of
//  lines in the vertical active area.
//  V_ACTIVE_START + V_ACTIVE_PERIOD should be
//  less than 2^LW_VI_V_IN (or 8192).
#define LWE24D_VI_CSI_PPA_V_ACTIVE_0_CSI_PPA_V_ACTIVE_PERIOD                     28:16

// This register defines the horizontal captured (active) area of the input video source with respect to
//  horizontal sync. (This is for CSI data.)

#define LWE24D_VI_CSI_PPB_H_ACTIVE_0                   (0x7c)
// Horizontal Active Start (offset to active)
//  This parameter specifies the number of
//  clock active edges from horizontal
//  sync active edge to the first horizontal
//  active pixel. If programmed to 0, the
//  first active line starts after the first
//  active clock edge following the horizontal
//  sync active edge.
#define LWE24D_VI_CSI_PPB_H_ACTIVE_0_CSI_PPB_H_ACTIVE_START                      12:0

// Horizontal Active Period
//  This parameter specifies the number of
//  pixels in the horizontal active area.
//  H_ACTIVE_START + H_ACTIVE_PERIOD should be
//  less than 2^LW_VI_H_IN (or 8192). This parameter
//  should be programmed with an even number
//  (bit 16 is ignored internally).
#define LWE24D_VI_CSI_PPB_H_ACTIVE_0_CSI_PPB_H_ACTIVE_PERIOD                     28:16

// This register defines the vertical captured (active) area of the input video source with respect to
//  vertical sync. (This is for CSI data.)

#define LWE24D_VI_CSI_PPB_V_ACTIVE_0                   (0x7d)
// Vertical Active Start (offset to active)
//  This parameter specifies the number of
//  horizontal sync active edges from vertical
//  sync active edge to the first vertical
//  active line. If programmed to 0, the
//  first active line starts after the first
//  horizontal sync active edge following
//  the vertical sync active edge.
#define LWE24D_VI_CSI_PPB_V_ACTIVE_0_CSI_PPB_V_ACTIVE_START                      12:0

// Vertical Active Period
//  This parameter specifies the number of
//  lines in the vertical active area.
//  V_ACTIVE_START + V_ACTIVE_PERIOD should be
//  less than 2^LW_VI_V_IN (or 8192).
#define LWE24D_VI_CSI_PPB_V_ACTIVE_0_CSI_PPB_V_ACTIVE_PERIOD                     28:16

// Used only with input from ISP: defines input image horizontal size in pixels

#define LWE24D_VI_ISP_H_ACTIVE_0                       (0x7e)
// Horizontal image size in pixels coming out of ISP.
// Must be an even number (bit 0 is ignored).
#define LWE24D_VI_ISP_H_ACTIVE_0_ISP_H_ACTIVE_PERIOD                     12:0

// Used only with input from ISP: defines input image vertical size in lines

#define LWE24D_VI_ISP_V_ACTIVE_0                       (0x7f)
// Vertical image size in lines coming out of ISP.
// Must be an even number (bit 0 is ignored).
#define LWE24D_VI_ISP_V_ACTIVE_0_ISP_V_ACTIVE_PERIOD                     12:0

// Stream raises ("safe to reprogram VI" raises)
// The I/O resources used by a data stream going through VI are indicated in STREAM_?_RESOURCE_DEFINE register.
// Once resources are set in this register,
// and after the start of the following picture,
// when ALL the stream's resources are done and idle processing that picture,
// a raise will be generated.
// It is then safe to reprogram VI's functional units ilwolved in processing that stream.
//
// Two simultaneous data streams are supported, and they don't have to be mutually exclusive.
//
// When no resources are indicated for a stream, no raise is generated.

// Field definition is: 0 = resource not used; 1 = resource used.
#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0                   (0x80)
#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_VIP_INPUT_1                 0:0
#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_VIP_INPUT_1_NOT_USED                      (0)
#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_VIP_INPUT_1_USED                  (1)

#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_HOST_INPUT_1                        1:1
#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_HOST_INPUT_1_NOT_USED                     (0)
#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_HOST_INPUT_1_USED                 (1)

#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_CSI_PPA_CROPPED_INPUT_1                     2:2
#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_CSI_PPA_CROPPED_INPUT_1_NOT_USED                  (0)
#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_CSI_PPA_CROPPED_INPUT_1_USED                      (1)

#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_CSI_PPA_UNCROPPED_INPUT_1                   3:3
#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_CSI_PPA_UNCROPPED_INPUT_1_NOT_USED                        (0)
#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_CSI_PPA_UNCROPPED_INPUT_1_USED                    (1)

#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_CSI_PPB_CROPPED_INPUT_1                     4:4
#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_CSI_PPB_CROPPED_INPUT_1_NOT_USED                  (0)
#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_CSI_PPB_CROPPED_INPUT_1_USED                      (1)

#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_CSI_PPB_UNCROPPED_INPUT_1                   5:5
#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_CSI_PPB_UNCROPPED_INPUT_1_NOT_USED                        (0)
#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_CSI_PPB_UNCROPPED_INPUT_1_USED                    (1)

#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_ISP_INPUT_1                 6:6
#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_ISP_INPUT_1_NOT_USED                      (0)
#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_ISP_INPUT_1_USED                  (1)

#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_FIRST_OUTPUT_1                      7:7
#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_FIRST_OUTPUT_1_NOT_USED                   (0)
#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_FIRST_OUTPUT_1_USED                       (1)

#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_SECOND_OUTPUT_1                     8:8
#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_SECOND_OUTPUT_1_NOT_USED                  (0)
#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_SECOND_OUTPUT_1_USED                      (1)

#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_HOST_DMA_VSYNC_OUTPUT_1                     9:9
#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_HOST_DMA_VSYNC_OUTPUT_1_NOT_USED                  (0)
#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_HOST_DMA_VSYNC_OUTPUT_1_USED                      (1)

#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_EPP_OUTPUT_1                        10:10
#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_EPP_OUTPUT_1_NOT_USED                     (0)
#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_EPP_OUTPUT_1_USED                 (1)

#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_HOST_DMA_BUFFER_OUTPUT_1                    11:11
#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_HOST_DMA_BUFFER_OUTPUT_1_NOT_USED                 (0)
#define LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0_HOST_DMA_BUFFER_OUTPUT_1_USED                     (1)


// Field definition is: 0 = resource not used; 1 = resource used.
#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0                   (0x81)
#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_VIP_INPUT_2                 0:0
#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_VIP_INPUT_2_NOT_USED                      (0)
#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_VIP_INPUT_2_USED                  (1)

#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_HOST_INPUT_2                        1:1
#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_HOST_INPUT_2_NOT_USED                     (0)
#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_HOST_INPUT_2_USED                 (1)

#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_CSI_PPA_CROPPED_INPUT_2                     2:2
#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_CSI_PPA_CROPPED_INPUT_2_NOT_USED                  (0)
#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_CSI_PPA_CROPPED_INPUT_2_USED                      (1)

#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_CSI_PPA_UNCROPPED_INPUT_2                   3:3
#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_CSI_PPA_UNCROPPED_INPUT_2_NOT_USED                        (0)
#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_CSI_PPA_UNCROPPED_INPUT_2_USED                    (1)

#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_CSI_PPB_CROPPED_INPUT_2                     4:4
#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_CSI_PPB_CROPPED_INPUT_2_NOT_USED                  (0)
#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_CSI_PPB_CROPPED_INPUT_2_USED                      (1)

#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_CSI_PPB_UNCROPPED_INPUT_2                   5:5
#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_CSI_PPB_UNCROPPED_INPUT_2_NOT_USED                        (0)
#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_CSI_PPB_UNCROPPED_INPUT_2_USED                    (1)

#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_ISP_INPUT_2                 6:6
#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_ISP_INPUT_2_NOT_USED                      (0)
#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_ISP_INPUT_2_USED                  (1)

#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_FIRST_OUTPUT_2                      7:7
#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_FIRST_OUTPUT_2_NOT_USED                   (0)
#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_FIRST_OUTPUT_2_USED                       (1)

#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_SECOND_OUTPUT_2                     8:8
#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_SECOND_OUTPUT_2_NOT_USED                  (0)
#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_SECOND_OUTPUT_2_USED                      (1)

#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_HOST_DMA_VSYNC_OUTPUT_2                     9:9
#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_HOST_DMA_VSYNC_OUTPUT_2_NOT_USED                  (0)
#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_HOST_DMA_VSYNC_OUTPUT_2_USED                      (1)

#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_EPP_OUTPUT_2                        10:10
#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_EPP_OUTPUT_2_NOT_USED                     (0)
#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_EPP_OUTPUT_2_USED                 (1)

#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_HOST_DMA_BUFFER_OUTPUT_2                    11:11
#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_HOST_DMA_BUFFER_OUTPUT_2_NOT_USED                 (0)
#define LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0_HOST_DMA_BUFFER_OUTPUT_2_USED                     (1)


// as defined by STREAM_1_RESOURCE_DEFINE register,
// become idle after the start of the following frame.
#define LWE24D_VI_RAISE_STREAM_1_DONE_0                        (0x82)
#define LWE24D_VI_RAISE_STREAM_1_DONE_0_RAISE_VECTOR_STREAM_1                    4:0

#define LWE24D_VI_RAISE_STREAM_1_DONE_0_RAISE_CHANNEL_STREAM_1                   19:16


// as defined by STREAM_2_RESOURCE_DEFINE register,
// become idle after the start of the following frame
#define LWE24D_VI_RAISE_STREAM_2_DONE_0                        (0x83)
#define LWE24D_VI_RAISE_STREAM_2_DONE_0_RAISE_VECTOR_STREAM_2                    4:0

#define LWE24D_VI_RAISE_STREAM_2_DONE_0_RAISE_CHANNEL_STREAM_2                   19:16

// ISDB-T tuner mode register set
//   tuner/demodulator mode.

#define LWE24D_VI_TS_MODE_0                    (0x84)
// This field indicates the global enable for ISDB-T protocol handling
#define LWE24D_VI_TS_MODE_0_ENABLE                       0:0
#define LWE24D_VI_TS_MODE_0_ENABLE_DISABLED                    (0)
#define LWE24D_VI_TS_MODE_0_ENABLE_ENABLED                     (1)

// This field determines if input data is in serial or parallel format
#define LWE24D_VI_TS_MODE_0_INPUT_MODE                   1:1
#define LWE24D_VI_TS_MODE_0_INPUT_MODE_PARALLEL                        (0)
#define LWE24D_VI_TS_MODE_0_INPUT_MODE_SERIAL                  (1)

// This field selected the pin configuration used for VD[1]  NONE:     TS_ERROR is tied to 0
//            TS_PSYNC is tied to 0
//  TS_ERROR: TS_ERROR is on VD[1]
//            TS_PSYNC is tied to 0
//  TS_PSYNC: TS_ERROR is tied to 0
//            TS_PSYNC is on VD[1]
#define LWE24D_VI_TS_MODE_0_PROTOCOL_SELECT                      3:2
#define LWE24D_VI_TS_MODE_0_PROTOCOL_SELECT_NONE                       (0)
#define LWE24D_VI_TS_MODE_0_PROTOCOL_SELECT_TS_ERROR                   (1)
#define LWE24D_VI_TS_MODE_0_PROTOCOL_SELECT_TS_PSYNC                   (2)
#define LWE24D_VI_TS_MODE_0_PROTOCOL_SELECT_RESERVED                   (3)

// This field selects the buffer flow control for the Write DMA RDMA:      The RDMA engine will release the buffers back to the WDMA
//            as the buffers are consumed
// NONE:      The VI will automatically release the buffer back to the
//            WMDA after each buffer ready is generated.
// CPU:       SW needs to write the TS_CPU_FLOW_CTL register to release
//            each buffer to the WDMA
#define LWE24D_VI_TS_MODE_0_FLOW_CONTROL_MODE                    5:4
#define LWE24D_VI_TS_MODE_0_FLOW_CONTROL_MODE_RDMA                     (0)
#define LWE24D_VI_TS_MODE_0_FLOW_CONTROL_MODE_NONE                     (1)
#define LWE24D_VI_TS_MODE_0_FLOW_CONTROL_MODE_CPU                      (2)
#define LWE24D_VI_TS_MODE_0_FLOW_CONTROL_MODE_RESERVED                 (3)


#define LWE24D_VI_TS_CONTROL_0                 (0x85)
// This field indicates the polarity of TS_VALID. Only has affect when TS_MODE.ENABLE == ENABLED    LOW indicates that the polarity of TS_VALID is active low.
//    HIGH indicates that the polarity of TS_VALID is active high.
#define LWE24D_VI_TS_CONTROL_0_VALID_POLARITY                    0:0
#define LWE24D_VI_TS_CONTROL_0_VALID_POLARITY_HIGH                     (0)
#define LWE24D_VI_TS_CONTROL_0_VALID_POLARITY_LOW                      (1)

#define LWE24D_VI_TS_CONTROL_0_PSYNC_POLARITY                    1:1
#define LWE24D_VI_TS_CONTROL_0_PSYNC_POLARITY_HIGH                     (0)
#define LWE24D_VI_TS_CONTROL_0_PSYNC_POLARITY_LOW                      (1)

#define LWE24D_VI_TS_CONTROL_0_ERROR_POLARITY                    2:2
#define LWE24D_VI_TS_CONTROL_0_ERROR_POLARITY_HIGH                     (0)
#define LWE24D_VI_TS_CONTROL_0_ERROR_POLARITY_LOW                      (1)

#define LWE24D_VI_TS_CONTROL_0_CLK_POLARITY                      3:3
#define LWE24D_VI_TS_CONTROL_0_CLK_POLARITY_HIGH                       (0)
#define LWE24D_VI_TS_CONTROL_0_CLK_POLARITY_LOW                        (1)

// This field defines how the START of packet condition is determined  PSYNC: PSYNC assertion rising edge
//  VALID: VALID assertion rising edge
//  BOTH:  PSYNC && VALID asserted rising edge
#define LWE24D_VI_TS_CONTROL_0_START_SELECT                      5:4
#define LWE24D_VI_TS_CONTROL_0_START_SELECT_RESERVED                   (0)
#define LWE24D_VI_TS_CONTROL_0_START_SELECT_PSYNC                      (1)
#define LWE24D_VI_TS_CONTROL_0_START_SELECT_VALID                      (2)
#define LWE24D_VI_TS_CONTROL_0_START_SELECT_BOTH                       (3)

// This field determines if VALID is used during BODY packet capture  IGNORE: the VALID signal is ignored during the capture
//  GATE: the VALID signal gates the capture of BODY data.
#define LWE24D_VI_TS_CONTROL_0_BODY_VALID_SELECT                 6:6
#define LWE24D_VI_TS_CONTROL_0_BODY_VALID_SELECT_IGNORE                        (0)
#define LWE24D_VI_TS_CONTROL_0_BODY_VALID_SELECT_GATE                  (1)

// This field determines is VI should store packets to memory that have been flagged as UPSTREAM_ERROR packets.
//  DISCARD: Do not store packets in memory
//  STORE:   Store UPSTREAM_ERROR packets in memory
#define LWE24D_VI_TS_CONTROL_0_STORE_UPSTREAM_ERROR_PKTS                 7:7
#define LWE24D_VI_TS_CONTROL_0_STORE_UPSTREAM_ERROR_PKTS_DISCARD                       (0)
#define LWE24D_VI_TS_CONTROL_0_STORE_UPSTREAM_ERROR_PKTS_STORE                 (1)

// This field stores the number of BODY bytes to capture (including PSYNC)
#define LWE24D_VI_TS_CONTROL_0_BODY_SIZE                 23:16

// This field stores the number of FEC bytes to catpure (after the BODY has been captured)
#define LWE24D_VI_TS_CONTROL_0_FEC_SIZE                  30:24


#define LWE24D_VI_TS_PACKET_COUNT_0                    (0x86)
// This field holds the current value of the received packet counter.  This counter increments
// in the presence of a new packet, regardless of whether it is flagged as an error
// The counter can be cleared by writing this register with 0's and can also
// be preloaded to any value by writing the preload value to the register.
#define LWE24D_VI_TS_PACKET_COUNT_0_TS_PACKET_COUNT_VALUE                        15:0

// This field is set to OVERFLOW when VALUE passes from 0xFFFF to 0x00000000000. It stays high until the CPU writes a zero to this bit to reset it.
#define LWE24D_VI_TS_PACKET_COUNT_0_TS_PACKET_COUNT_VALUE_OVERFLOW                       16:16
#define LWE24D_VI_TS_PACKET_COUNT_0_TS_PACKET_COUNT_VALUE_OVERFLOW_NONE                        (0)
#define LWE24D_VI_TS_PACKET_COUNT_0_TS_PACKET_COUNT_VALUE_OVERFLOW_OVERFLOW                    (1)


#define LWE24D_VI_TS_ERROR_COUNT_0                     (0x87)
// This field holds the current value of the error packet counter.  This counter increments in the
// presence of a packet flagged as error (see TS_ERROR)0000 or a detected protocol violation.
// The counter can be cleared by writing this register with 0's and can also
// be preloaded to any value by writing the preload value to the register.
#define LWE24D_VI_TS_ERROR_COUNT_0_TS_ERROR_COUNT_VALUE                  15:0

// This field is set to OVEFLOW when VALUE passes from 0xFFFF to 0x00000000000. It stays high until the CPU writes a zero to this bit to reset it.
#define LWE24D_VI_TS_ERROR_COUNT_0_TS_ERROR_COUNT_VALUE_OVERFLOW                 16:16
#define LWE24D_VI_TS_ERROR_COUNT_0_TS_ERROR_COUNT_VALUE_OVERFLOW_NONE                  (0)
#define LWE24D_VI_TS_ERROR_COUNT_0_TS_ERROR_COUNT_VALUE_OVERFLOW_OVERFLOW                      (1)


#define LWE24D_VI_TS_CPU_FLOW_CTL_0                    (0x88)
// Used only when the FLOW_CONTROL_MODE register is set to CPU
// SW must write this register to release each buffer back to
// WDMA.  Failure to write this register when buffers are
// consumed will result in the WDMA stalling when it consumes all
// allocated/free buffers.
#define LWE24D_VI_TS_CPU_FLOW_CTL_0_BUFFER_RELEASE                       0:0

// We are using HOST_DMA_WRITE_BUFFER.BUFFER_SIZE (bytes) to hold the number of bytes in a buffer for ISDB-T mode.

// This feature was introduced in SC17,
// and represents an alternative value to using
// VB0_BUFFER_STRIDE_C.
#define LWE24D_VI_VB0_CHROMA_BUFFER_STRIDE_FIRST_0                     (0x89)
// Chroma buffer stride in bytes
// 30 reserved
#define LWE24D_VI_VB0_CHROMA_BUFFER_STRIDE_FIRST_0_VB0_CHROMA_BUFFER_STRIDE                      29:0

// select type of Chroma buffer stride: 0 = Use VB0_BUFFER_STRIDE_C, deriving chroma
// buffer stride from luma buffer stride
// (default and backward compatible to SC15).
// 1 = Use VB0_CHROMA_BUFFER_STRIDE.
#define LWE24D_VI_VB0_CHROMA_BUFFER_STRIDE_FIRST_0_VB0_CHROMA_BUFFER_STRIDE_SELECT                       31:31
#define LWE24D_VI_VB0_CHROMA_BUFFER_STRIDE_FIRST_0_VB0_CHROMA_BUFFER_STRIDE_SELECT_RATIO                       (0)
#define LWE24D_VI_VB0_CHROMA_BUFFER_STRIDE_FIRST_0_VB0_CHROMA_BUFFER_STRIDE_SELECT_VALUE                       (1)


#define LWE24D_VI_VB0_CHROMA_LINE_STRIDE_FIRST_0                       (0x8a)
// Video Buffer Set 0 chroma horizontal size
//  This parameter specifies the chroma line stride
//  (in pixels) for lines in the video buffer
//  set 0.
//  this parameter
//  must be programmed as multiple of 4 pixels
//  (bits 1-0 are ignored).
#define LWE24D_VI_VB0_CHROMA_LINE_STRIDE_FIRST_0_VB0_CHROMA_H_SIZE_1                     12:0

// select type of Chroma line stride: 0 = Use VB0_H_SIZE_1, deriving chroma line stride from luma line stride (default and backward compatible to SC15).
// 1 = Use VB0_CHROMA_H_SIZE_1.
#define LWE24D_VI_VB0_CHROMA_LINE_STRIDE_FIRST_0_VB0_CHROMA_LINE_STRIDE_SELECT                   31:31
#define LWE24D_VI_VB0_CHROMA_LINE_STRIDE_FIRST_0_VB0_CHROMA_LINE_STRIDE_SELECT_RATIO                   (0)
#define LWE24D_VI_VB0_CHROMA_LINE_STRIDE_FIRST_0_VB0_CHROMA_LINE_STRIDE_SELECT_VALUE                   (1)

// this reg. is used for VI2EPP syncpt only.
// VI will based on num_lines = frame_height/EPP_NUM_OF_BUFFER_PER_FRAME,
// send vi2epp_trigger for every num_lines

#define LWE24D_VI_EPP_LINES_PER_BUFFER_0                       (0x8b)
// maximum 256 buffers per frame.
// linesPerBuffer = FLOOR(eppLineCount/eppBufferCount)
// linesPerBuffer must be > 2
// eppLineCount must take into account any cropping in EPP.
#define LWE24D_VI_EPP_LINES_PER_BUFFER_0_LINES_PER_BUFFER                        12:0


// BUFFER_COUNTER by 1
#define LWE24D_VI_BUFFER_RELEASE_OUTPUT1_0                     (0x8c)
#define LWE24D_VI_BUFFER_RELEASE_OUTPUT1_0_BUFFER_RELEASE_OUTPUT1                        0:0


#define LWE24D_VI_BUFFER_RELEASE_OUTPUT2_0                     (0x8d)
#define LWE24D_VI_BUFFER_RELEASE_OUTPUT2_0_BUFFER_RELEASE_OUTPUT2                        0:0


#define LWE24D_VI_DEBUG_FLOW_CONTROL_COUNTER_OUTPUT1_0                 (0x8e)
#define LWE24D_VI_DEBUG_FLOW_CONTROL_COUNTER_OUTPUT1_0_BUFFER_COUNT_OUTPUT1                      7:0


#define LWE24D_VI_DEBUG_FLOW_CONTROL_COUNTER_OUTPUT2_0                 (0x8f)
#define LWE24D_VI_DEBUG_FLOW_CONTROL_COUNTER_OUTPUT2_0_BUFFER_COUNT_OUTPUT2                      7:0

// register for SW to write to terminate MC BW

// MC on BW operation in FIRST output.
#define LWE24D_VI_TERMINATE_BW_FIRST_0                 (0x90)
#define LWE24D_VI_TERMINATE_BW_FIRST_0_TERMINATE_FIRST_BW                        0:0


// MC on BW operationn in SECOND output.
#define LWE24D_VI_TERMINATE_BW_SECOND_0                        (0x91)
#define LWE24D_VI_TERMINATE_BW_SECOND_0_TERMINATE_SECOND_BW                      0:0

// Memory Controller Tiling definitions
//
//
//  To enable tiling for a buffer in your module you'll want to include
//  this spec file and then make use of either the ADD_TILE_MODE_REG_SPEC
//  or ADD_TILE_MODE_REG_FIELD_SPEC macro.
//
//  For the ADD_TILE_MODE_REG_SPEC macro, the regp arg is added to the
//  register name as a prefix to match the names of the other registers
//  for this buffer. The fldp is the field name prefix to make the name
//  unique so it works with arreggen generated reg blocks (e.g.):
//
//      // specify how addressing should occur for IB0 buffer
//      ADD_TILE_MODE_REG_SPEC(IB0, IB0);
//
//  There's also a REG_RW_SPEC version, if you need to specify a special
//  flag (e.g. rws for shadow, or rwt for trigger).
//  
//  For the ADD_TILE_MODE_REG_FIELD_SPEC macro, the fldp is the field
//  name prefix and bitpos arg describes the starting bit position for
//  this field within another register.
//
//  Like the register version, there's a REG_RW_FIELD_SPEC version if
//  you need to set explicit bits other than "rw".
//
//  Note: this requires having at least LW_MC_TILE_MODEWIDTH bits of
//  space available after bitpos (e.g.) in the register:
//
//      ADD_TILE_MODE_REG_FIELD_SPEC(REF, 16)   // This parameter specifies how addressing
//                                              // for the REF buffer should occur
//

#define LWE24D_VI_VB0_FIRST_BUFFER_ADDR_MODE_0                 (0x92)
#define LWE24D_VI_VB0_FIRST_BUFFER_ADDR_MODE_0_Y1_TILE_MODE                      0:0
#define LWE24D_VI_VB0_FIRST_BUFFER_ADDR_MODE_0_Y1_TILE_MODE_LINEAR                     (0)
#define LWE24D_VI_VB0_FIRST_BUFFER_ADDR_MODE_0_Y1_TILE_MODE_TILED                      (1)

#define LWE24D_VI_VB0_FIRST_BUFFER_ADDR_MODE_0_UV1_TILE_MODE                     8:8
#define LWE24D_VI_VB0_FIRST_BUFFER_ADDR_MODE_0_UV1_TILE_MODE_LINEAR                    (0)
#define LWE24D_VI_VB0_FIRST_BUFFER_ADDR_MODE_0_UV1_TILE_MODE_TILED                     (1)


#define LWE24D_VI_VB0_SECOND_BUFFER_ADDR_MODE_0                        (0x93)
#define LWE24D_VI_VB0_SECOND_BUFFER_ADDR_MODE_0_Y2_TILE_MODE                     0:0
#define LWE24D_VI_VB0_SECOND_BUFFER_ADDR_MODE_0_Y2_TILE_MODE_LINEAR                    (0)
#define LWE24D_VI_VB0_SECOND_BUFFER_ADDR_MODE_0_Y2_TILE_MODE_TILED                     (1)

//VIP Pattern Generator:
// The VIP pattern generator is new as of AP15.  When enabled, it overrides the inputs from
// the attached camera with internally generated pattern data, hsyncs, and vsyncs.  The
// purpose of the pattern generator is to facilitate regression testing of the VI driver and
// hardware without constraining the board level design.
//
// The pattern generator logic runs on the pd2vi_clock domain.  See the clock controller spec
// for information on how to enable a loopback from the vi_sensor clock to the pd2vi_clock.
//
// The user must program the pattern width, pattern height, and bayer select registers prior to
// enabling the pattern generator.  It is  illegal to change the values of those registers
// without first disabling the pattern generator.
//
// The pattern generator has no concept of blanking time.  The width and the height of the
// pattern should correspond to the full hblank+hactive and vblank+vactive
//

// bits[13:0] are reserved for
// VIP Pattern Gen (Pattern Width)
#define LWE24D_VI_RESERVE_0_0                  (0x94)
//  Program to *one less* than the desired
#define LWE24D_VI_RESERVE_0_0_nc_RESERVE_0_0                     3:0

//  pattern width in clocks. (note that
#define LWE24D_VI_RESERVE_0_0_nc_RESERVE_0_1                     7:4

//  there are 2 clocker per pixel for YUV422)
#define LWE24D_VI_RESERVE_0_0_nc_RESERVE_0_2                     11:8

#define LWE24D_VI_RESERVE_0_0_nc_RESERVE_0_3                     15:12


// bits[13:0] are reserved for
// VIP Pattern Gen (Pattern Height)
#define LWE24D_VI_RESERVE_1_0                  (0x95)
//  Program to *one less* than the desired
#define LWE24D_VI_RESERVE_1_0_nc_RESERVE_1_0                     3:0

//  pattern height in lines
#define LWE24D_VI_RESERVE_1_0_nc_RESERVE_1_1                     7:4

#define LWE24D_VI_RESERVE_1_0_nc_RESERVE_1_2                     11:8

#define LWE24D_VI_RESERVE_1_0_nc_RESERVE_1_3                     15:12


// bit 0 is reserved for VIP Pattern Gen Enable
// bit 1 is reserved for VIP Pattern Gen BayerSelect
#define LWE24D_VI_RESERVE_2_0                  (0x96)
// 1 for BAYER pattern and 0 for YUV pattern
#define LWE24D_VI_RESERVE_2_0_nc_RESERVE_2_0                     3:0

#define LWE24D_VI_RESERVE_2_0_nc_RESERVE_2_1                     7:4

#define LWE24D_VI_RESERVE_2_0_nc_RESERVE_2_2                     11:8

#define LWE24D_VI_RESERVE_2_0_nc_RESERVE_2_3                     15:12


#define LWE24D_VI_RESERVE_3_0                  (0x97)
#define LWE24D_VI_RESERVE_3_0_nc_RESERVE_3_0                     3:0

#define LWE24D_VI_RESERVE_3_0_nc_RESERVE_3_1                     7:4

#define LWE24D_VI_RESERVE_3_0_nc_RESERVE_3_2                     11:8

#define LWE24D_VI_RESERVE_3_0_nc_RESERVE_3_3                     15:12


#define LWE24D_VI_RESERVE_4_0                  (0x98)
#define LWE24D_VI_RESERVE_4_0_nc_RESERVE_4_0                     3:0

#define LWE24D_VI_RESERVE_4_0_nc_RESERVE_4_1                     7:4

#define LWE24D_VI_RESERVE_4_0_nc_RESERVE_4_2                     11:8

#define LWE24D_VI_RESERVE_4_0_nc_RESERVE_4_3                     15:12

// Memory Client Interface Hysteresis Registers
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

#define LWE24D_VI_MCCIF_VIRUV_HYST_0                   (0x99)
#define LWE24D_VI_MCCIF_VIRUV_HYST_0_CBR_VIRUV2MC_HYST_REQ_TM                    7:0

#define LWE24D_VI_MCCIF_VIRUV_HYST_0_CBR_VIRUV2MC_DHYST_TM                       15:8

#define LWE24D_VI_MCCIF_VIRUV_HYST_0_CBR_VIRUV2MC_DHYST_TH                       23:16

#define LWE24D_VI_MCCIF_VIRUV_HYST_0_CBR_VIRUV2MC_HYST_TM                        27:24

#define LWE24D_VI_MCCIF_VIRUV_HYST_0_CBR_VIRUV2MC_HYST_REQ_TH                    30:28

#define LWE24D_VI_MCCIF_VIRUV_HYST_0_CBR_VIRUV2MC_HYST_EN                        31:31
#define LWE24D_VI_MCCIF_VIRUV_HYST_0_CBR_VIRUV2MC_HYST_EN_ENABLE                       (1)
#define LWE24D_VI_MCCIF_VIRUV_HYST_0_CBR_VIRUV2MC_HYST_EN_DISABLE                      (0)

// Memory Client Hysteresis Control Register
// This register exists only for clients with hysteresis.
// BUG 505006: Hysteresis configuration can only be updated when memory traffic is idle.
// HYST_EN can be used to turn on or off the hysteresis logic.
// HYST_REQ_TH is the threshold of pending requests required
//   before allowing them to pass through
//   (overriden after HYST_REQ_TM cycles).

#define LWE24D_VI_MCCIF_VIWSB_HYST_0                   (0x9a)
#define LWE24D_VI_MCCIF_VIWSB_HYST_0_CBW_VIWSB2MC_HYST_REQ_TM                    11:0

#define LWE24D_VI_MCCIF_VIWSB_HYST_0_CBW_VIWSB2MC_HYST_REQ_TH                    30:28

#define LWE24D_VI_MCCIF_VIWSB_HYST_0_CBW_VIWSB2MC_HYST_EN                        31:31
#define LWE24D_VI_MCCIF_VIWSB_HYST_0_CBW_VIWSB2MC_HYST_EN_ENABLE                       (1)
#define LWE24D_VI_MCCIF_VIWSB_HYST_0_CBW_VIWSB2MC_HYST_EN_DISABLE                      (0)

// Memory Client Hysteresis Control Register
// This register exists only for clients with hysteresis.
// BUG 505006: Hysteresis configuration can only be updated when memory traffic is idle.
// HYST_EN can be used to turn on or off the hysteresis logic.
// HYST_REQ_TH is the threshold of pending requests required
//   before allowing them to pass through
//   (overriden after HYST_REQ_TM cycles).

#define LWE24D_VI_MCCIF_VIWU_HYST_0                    (0x9b)
#define LWE24D_VI_MCCIF_VIWU_HYST_0_CBW_VIWU2MC_HYST_REQ_TM                      11:0

#define LWE24D_VI_MCCIF_VIWU_HYST_0_CBW_VIWU2MC_HYST_REQ_TH                      30:28

#define LWE24D_VI_MCCIF_VIWU_HYST_0_CBW_VIWU2MC_HYST_EN                  31:31
#define LWE24D_VI_MCCIF_VIWU_HYST_0_CBW_VIWU2MC_HYST_EN_ENABLE                 (1)
#define LWE24D_VI_MCCIF_VIWU_HYST_0_CBW_VIWU2MC_HYST_EN_DISABLE                        (0)

// Memory Client Hysteresis Control Register
// This register exists only for clients with hysteresis.
// BUG 505006: Hysteresis configuration can only be updated when memory traffic is idle.
// HYST_EN can be used to turn on or off the hysteresis logic.
// HYST_REQ_TH is the threshold of pending requests required
//   before allowing them to pass through
//   (overriden after HYST_REQ_TM cycles).

#define LWE24D_VI_MCCIF_VIWV_HYST_0                    (0x9c)
#define LWE24D_VI_MCCIF_VIWV_HYST_0_CBW_VIWV2MC_HYST_REQ_TM                      11:0

#define LWE24D_VI_MCCIF_VIWV_HYST_0_CBW_VIWV2MC_HYST_REQ_TH                      30:28

#define LWE24D_VI_MCCIF_VIWV_HYST_0_CBW_VIWV2MC_HYST_EN                  31:31
#define LWE24D_VI_MCCIF_VIWV_HYST_0_CBW_VIWV2MC_HYST_EN_ENABLE                 (1)
#define LWE24D_VI_MCCIF_VIWV_HYST_0_CBW_VIWV2MC_HYST_EN_DISABLE                        (0)

// Memory Client Hysteresis Control Register
// This register exists only for clients with hysteresis.
// BUG 505006: Hysteresis configuration can only be updated when memory traffic is idle.
// HYST_EN can be used to turn on or off the hysteresis logic.
// HYST_REQ_TH is the threshold of pending requests required
//   before allowing them to pass through
//   (overriden after HYST_REQ_TM cycles).

#define LWE24D_VI_MCCIF_VIWY_HYST_0                    (0x9d)
#define LWE24D_VI_MCCIF_VIWY_HYST_0_CBW_VIWY2MC_HYST_REQ_TM                      11:0

#define LWE24D_VI_MCCIF_VIWY_HYST_0_CBW_VIWY2MC_HYST_REQ_TH                      30:28

#define LWE24D_VI_MCCIF_VIWY_HYST_0_CBW_VIWY2MC_HYST_EN                  31:31
#define LWE24D_VI_MCCIF_VIWY_HYST_0_CBW_VIWY2MC_HYST_EN_ENABLE                 (1)
#define LWE24D_VI_MCCIF_VIWY_HYST_0_CBW_VIWY2MC_HYST_EN_DISABLE                        (0)

// CSI register spec
// CSI (MIPI Camera Serial Interface) register definition

// Register CSI_VI_INPUT_STREAM_CONTROL_0  // VI Input Stream Control
#define LWE24D_CSI_VI_INPUT_STREAM_CONTROL_0                   (0x200)
// VIP Start Frame Generation Don't use vi2csi_vip_vsync to generate start frame
// (SF), or end frame (EF) markers in the pixel parser
// output stream.
#define LWE24D_CSI_VI_INPUT_STREAM_CONTROL_0_VIP_SF_GEN                  7:7
#define LWE24D_CSI_VI_INPUT_STREAM_CONTROL_0_VIP_SF_GEN_VSYNC_SF                       (0)    // // Pulses on vi2csi_vip_vsync will be used to
// generate start frame (SF) and end frame (EF) markers
// in the pixel parser output stream.
// In AP15, only payload_only mode is supported in
// the VIP input stream path, and this fields may 
// always be programmed to VSYNC_SF.

#define LWE24D_CSI_VI_INPUT_STREAM_CONTROL_0_VIP_SF_GEN_NO_VSYNC_SF                    (1)


// reserved for additional VI Input Stream control register
// in case it is needed in the future

// Register CSI_HOST_INPUT_STREAM_CONTROL_0  // Host Input Stream Control
#define LWE24D_CSI_HOST_INPUT_STREAM_CONTROL_0                 (0x202)
// Host Data Format Data written to Y_FIFO_WRITE port should be in CSI
// packet format. To indicate end of packet a 1 should
// be written to HOST_END_OF_PACKET. A 1 should also be
// written to HOST_END_OF_PACKET before writing the first
// word of packet data to Y_FIFO_WRITE.
#define LWE24D_CSI_HOST_INPUT_STREAM_CONTROL_0_HOST_DATA_FORMAT                  3:0
#define LWE24D_CSI_HOST_INPUT_STREAM_CONTROL_0_HOST_DATA_FORMAT_PAYLOAD_ONLY                   (0)    // // Data written to Y_FIFO_WRITE port should be
// CSI line payload data only (no header, no footer,
// and  no short packets). A value of 1 should not
// be written to HOST_END_OF_PACKET (end of packet
// pulse only gets generated when a 1 is written to
// this bit). 
// First line will be indicated when one of the pixel
// parsers is first enabled with its 
// CSI_PPA/B_STREAM_SOURCE set to "HOST".
// The values in the following PIXEL_STREAM_A/B_CONTROL0 
// fields, for the pixel parser that is receiving host
// data, will be ignored;
// CSI_PPA/B_PACKET_HEADER overridden with "NOT_SENT",
// CSI_PPA/B_DATA_IDENTIFIER overridden with "DISABLED",
// CSI_PPA/B_WORD_COUNT_SELECT overridden with "REGISTER".
// CSI_PPA/B_CRC_CHECK overridden with "DISABLE",
// CSI_PPA/B_VIRTUAL_CHANNEL_ID,
// CSI_PPA/B_EMBEDDED_DATA_OPTIONS, and
// CSI_PPA/B_HEADER_EC_ENABLE.
// CSI_PPA/B_DATA_TYPE should be programmed with the 
// 6 bit data type that is to be used to interpret the
// the number of bytes per line.

#define LWE24D_CSI_HOST_INPUT_STREAM_CONTROL_0_HOST_DATA_FORMAT_PACKETS                        (1)

// Host Start Frame Generation Don't use CSI Host Line counter to generate start, or
// End, of Frame control outputs. This setting should only
// be used if HOST_DATA_FORMAT is set to PACKETS, and the
// Host data stream has frame sync packets.
#define LWE24D_CSI_HOST_INPUT_STREAM_CONTROL_0_HOST_SF_GEN                       7:7
#define LWE24D_CSI_HOST_INPUT_STREAM_CONTROL_0_HOST_SF_GEN_LINE_COUNTER                        (0)    // // CSI Host Line counter will be used to generate Frame
// start and end control. To signal the start of the first
// frame the pixel parser will send a SF control, and
// signal start of frame mark, when it is first enabled
// with Host as its source. This setting should be used 
// when HOST_DATA_FORMAT is set to PAYLOAD_ONLY.

#define LWE24D_CSI_HOST_INPUT_STREAM_CONTROL_0_HOST_SF_GEN_SHORT_PACKETS                       (1)

// Writing this bit with a 1 indicates End of Packet,
// when CSI Host data is being received in Packet Format.
// In Packet Format vi2csi_host_hsync is not used to 
// indicate beginning of packet.
#define LWE24D_CSI_HOST_INPUT_STREAM_CONTROL_0_HOST_END_OF_PACKET                        8:8

// Host Frame Height
// Specifies the height of the host frame when the host
// is supplying CSI format payload only data to one of 
// the CSI pixel parsers.
// Programmed Value = number of lines - 1
#define LWE24D_CSI_HOST_INPUT_STREAM_CONTROL_0_HOST_FRAME_HEIGHT                 28:16


// reserved for additional Host Input Stream control register
// in case it is needed in the future

// Register CSI_INPUT_STREAM_A_CONTROL_0  // CSI Input Stream A Control
#define LWE24D_CSI_INPUT_STREAM_A_CONTROL_0                    (0x204)
// CSI-A Data Lane
//   0= 1 data lane
//   1= 2 data lanes
//   2= 3 data lanes (not supported on SC17 & SC25)
//   3= 4 data lanes (not supported on SC17 & SC25)
#define LWE24D_CSI_INPUT_STREAM_A_CONTROL_0_CSI_A_DATA_LANE                      1:0

// Enables skip packet threshold feature. Skip packet feature is enabled.
#define LWE24D_CSI_INPUT_STREAM_A_CONTROL_0_CSI_A_SKIP_PACKET_THRESHOLD_ENABLE                   4:4
#define LWE24D_CSI_INPUT_STREAM_A_CONTROL_0_CSI_A_SKIP_PACKET_THRESHOLD_ENABLE_DISABLE                 (0)    // // Skip packet feature is disabled.     

#define LWE24D_CSI_INPUT_STREAM_A_CONTROL_0_CSI_A_SKIP_PACKET_THRESHOLD_ENABLE_ENABLE                  (1)

// CSI-A Skip Packet Threshold
//  This value is compared against the internal
//  FIFO that buffer the input streams. A packet
//  will be skipped (discarded) if the pixel
//  stream processor is busy (probably due to
//  padding process of a short line) and the
//  number of entries in the internal FIFO
//  exceeds this threshold value. Note that
//  each entry in the internal FIFO buffer is
//  four bytes.
//  To turn off this feature, set the value
//  to its maximum value (all ones).
#define LWE24D_CSI_INPUT_STREAM_A_CONTROL_0_CSI_A_SKIP_PACKET_THRESHOLD                  23:16


// reserved for additional Input Stream control register
// in case it is needed in the future

// Register CSI_PIXEL_STREAM_A_CONTROL0_0  // CSI Pixel Stream A Control 0
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0                   (0x206)
// CSI Pixel Parser A Stream Source   Host
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_STREAM_SOURCE                       2:0
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_STREAM_SOURCE_CSI_A                       (0)    // //   CSI Interface A

#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_STREAM_SOURCE_CSI_B                       (1)    // //   CSI Interface B

#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_STREAM_SOURCE_VI_PORT                     (6)    // //   VI port

#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_STREAM_SOURCE_HOST                        (7)

// CSI Pixel Parser A Packet Header processing
//  This specifies whether packet header is
//  sent in the beginning of packet or not.      Packet header is sent.
//      This setting should be used if the
//      stream source is CSI Interface A or B.
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_PACKET_HEADER                       4:4
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_PACKET_HEADER_NOT_SENT                    (0)    // //      Packet header is not sent.
//      This setting should not be used if the
//      stream source is CSI Interface A or B.
//      Unless CSI-A, or CSI-B, is operating in a
//      stream capture debug mode.
//      In this case, CSI_PPA_DATA_TYPE specifies
//      the stream data format and the number
//      of bytes per line/packet is
//      specified by CSI_PPA_WORD_COUNT.
//      This implies that a packet footer
//      is also not sent.  In this case, no 
//      packet footer CRC check should be performed.

#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_PACKET_HEADER_SENT                        (1)

// CSI Pixel Parser A Data Identifier (DI) byte processing
//  This parameter is effective only if packet
//  header is sent as part of the stream.      Enabled  - Data Identifier byte in
//      packet header should be compared against
//      the CSI_PPA_DATA_TYPE and the
//      CSI_PPA_VIRTUAL_CHANNEL_ID.
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_DATA_IDENTIFIER                     5:5
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_DATA_IDENTIFIER_DISABLED                  (0)    // //      Disabled - Data Identifier byte in
//      packet header should be ignored
//      (not checked against CSI_PPA_DATA_TYPE
//      and against CSI_PPA_VIRTUAL_CHANNEL_ID).
//      In this case, CSI_PPA_DATA_TYPE specifies
//      the stream data format.

#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_DATA_IDENTIFIER_ENABLED                   (1)

// CSI Pixel Parser A Word Count Select
//  This parameter is effective only if packet
//  header is sent as part of the stream.      The number of bytes per line is to be
//      extracted from Word Count field in the
//      packet header. Note that if the serial
//      link is not error free, programming this
//      bit to HEADER may be dangerous because 
//      the word count information in the header 
//      may be corrupted. 
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_WORD_COUNT_SELECT                   6:6
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_WORD_COUNT_SELECT_REGISTER                        (0)    // //      Word Count in packet header is ignored
//      and the number of bytes per line/packet
//      is specified by CSI_PPA_WORD_COUNT. Payload
//      CRC check will not be valid if the word
//      than the count in the packet header.
//      It is recommended to always program
//      this bit to REGISTER and always program
//      CSI_PPA_WORD_COUNT.

#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_WORD_COUNT_SELECT_HEADER                  (1)

// CSI Pixel Parser A Data CRC Check
//  This parameter specifies whether the last
//  2 bytes of packet should be treated as
//  CRC checksum and used to perform CRC check
//  on the payload data. Note that in case there
//  are 2 bytes of data CRC at the end of the
//  packet, the packet word count does not
//  include the CRC bytes.      Data CRC Check is enabled.
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_CRC_CHECK                   7:7
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_CRC_CHECK_DISABLE                 (0)    // //      Data CRC Check is disabled regardless
//      of whether there are CRC checksum at
//      the end of the packet.

#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_CRC_CHECK_ENABLE                  (1)

// CSI Pixel Parser A Data Type  This is CSI compatible data type as defined
//  in CSI specification. If the source stream
//  contains packet headers this value can be compared
//  to the CSI Data Type value in the 6 LSB of the
//  CSI Data Identifier (DI) byte. If the source stream
//  doesn't contain packet headers, or CSI_PPA_DATA_IDENTIFIER
//  is DISABLED, this value will be used to determine how
//  the stream will be colwerted to pixels.
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_DATA_TYPE                   13:8
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_DATA_TYPE_YUV420_8                        (24)
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_DATA_TYPE_YUV420_10                       (25)
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_DATA_TYPE_LEG_YUV420_8                    (26)
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_DATA_TYPE_YUV420CSPS_8                    (28)
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_DATA_TYPE_YUV420CSPS_10                   (29)
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_DATA_TYPE_YUV422_8                        (30)
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_DATA_TYPE_YUV422_10                       (31)
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_DATA_TYPE_RGB444                  (32)
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_DATA_TYPE_RGB555                  (33)
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_DATA_TYPE_RGB565                  (34)
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_DATA_TYPE_RGB666                  (35)
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_DATA_TYPE_RGB888                  (36)
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_DATA_TYPE_RAW6                    (40)
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_DATA_TYPE_RAW7                    (41)
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_DATA_TYPE_RAW8                    (42)
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_DATA_TYPE_RAW10                   (43)
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_DATA_TYPE_RAW12                   (44)
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_DATA_TYPE_RAW14                   (45)
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_DATA_TYPE_ARB_DT1                 (48)
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_DATA_TYPE_ARB_DT2                 (49)
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_DATA_TYPE_ARB_DT3                 (50)
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_DATA_TYPE_ARB_DT4                 (51)

// CSI Pixel Parser A Virtual Channel Identifier  
//  This is CSI compatible virtual channel
//  identifier as defined in CSI specification.
//  If the source stream contains packet headers
//  and CSI_PPA_DATA_IDENTIFIER is ENABLED this
//  value will be compared to the CSI Virtual
//  Channel Identifier value in the 2 MSB of the
//  CSI Data Identifier (DI) byte. This value will
//  be ignored if the source stream doesn't contain
//  packet headers, or CSI_PPA_DATA_IDENTIFIER is 
//  DISABLED, then this value will be ignored.
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_VIRTUAL_CHANNEL_ID                  15:14
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_VIRTUAL_CHANNEL_ID_ONE                    (0)
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_VIRTUAL_CHANNEL_ID_TWO                    (1)
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_VIRTUAL_CHANNEL_ID_THREE                  (2)
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_VIRTUAL_CHANNEL_ID_FOUR                   (3)

// CSI Pixel Parser A Output Format Options
//  This parameter specifies options for output data 
//  format.       Output for storing RAW data to memory through
//       ISP. Undefined LS color bits for RGB_666, 
//       RGB_565, RGB_555, and RGB_444, will be zeroed.
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_OUTPUT_FORMAT_OPTIONS                       19:16
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_OUTPUT_FORMAT_OPTIONS_ARBITRARY                   (0)    // //       Output as 8-bit arbitrary data stream
//       This may be used for compressed JPEG stream

#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_OUTPUT_FORMAT_OPTIONS_PIXEL                       (1)    // //       Output the normal 1 pixel/clock. Undefined 
//       LS color bits for RGB_666, RGB_565, RGB_555,
//       and RGB_444, will be zeroed.

#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_OUTPUT_FORMAT_OPTIONS_PIXEL_REP                   (2)    // //       Same as PIXEL except MS color bits, for RGB_666, 
//       RGB_565, RGB_555, and RGB_444, will be 
//       replicated to their undefined LS bits.

#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_OUTPUT_FORMAT_OPTIONS_STORE                       (3)

// CSI Pixel Parser A Embedded Data Options 
//  This specifies how to deal with embedded
//  data within the specified input stream
//  assuming that the CSI_PPA_DATA_TYPE is not
//  embedded data and assuming that embedded
//  data is not already processed by other
//  CSI pixel stream processor.       output embedded data as 8-bpp arbitrary
//       data stream.
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_EMBEDDED_DATA_OPTIONS                       21:20
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_EMBEDDED_DATA_OPTIONS_DISCARD                     (0)    // //       discard (throw away) embedded data

#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_EMBEDDED_DATA_OPTIONS_EMBEDDED                    (1)

// CSI Pixel Parser A Pad Short Line
//  This specifies how to deal with shorter than
//  expected line (the number of bytes received
//  is less than the specified word count)       short line is not padded (will output
//       less pixels than expected).
//       This option is not recommended and may
//       cause other modules that receives CSI
//       output stream to hang up.
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_PAD_SHORT_LINE                      25:24
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_PAD_SHORT_LINE_PAD0S                      (0)    // //       short line is padded by pixel of zeros
//       such that the expected number of output
//       pixels is correct. Due to the time
//       required to do the padding, subsequent
//       line packet maybe discarded and
//       therefore may cause a short frame
//       (total number of lines per frame is
//       less than expected).

#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_PAD_SHORT_LINE_PAD1S                      (1)    // //       short line is padded by pixel of ones
//       such that the expected number of output
//       pixels is correct. Due to the time
//       required to do the padding, subsequent
//       line packet maybe discarded and
//       therefore may cause a short frame
//       (total number of lines per frame is
//       less than expected).

#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_PAD_SHORT_LINE_NOPAD                      (2)

// CSI Pixel Parser A Packet Header Error Correction Enable
//  This parameter specifies whether single bit
//  errors in the packet header will be
//  automatically corrected, or not.    Single bit errors in the header will not
//    be corrected. Header ECC check will still
//    set header ECC status bits and the packet
//    will be processed by Pixel Parser A. DISABLE
//    should not be used when processing interleaved
//    streams (Same stream going to both PPA and PPB).
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_HEADER_EC_ENABLE                    27:27
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_HEADER_EC_ENABLE_ENABLE                   (0)    // //    Single bit errors in the header will be
//    automatically corrected.

#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_HEADER_EC_ENABLE_DISABLE                  (1)

// CSI Pixel Parser A Pad Frame
//  This specifies how to deal with frames that are
//  shorter (fewer lines) that expected. Short frames
//  are usually caused by line packets being dropped
//  because of packet errors. Expected frame height is
//  specified in PPA_EXP_FRAME_HEIGHT. To do padding the
//  number of input bytes in each line's payload.  Short frames will not be padded out.   
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_PAD_FRAME                   29:28
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_PAD_FRAME_PAD0S                   (0)    // //  Lines of all zeros will be used to pad out frames
//  that are shorter than expected height. 
//  PPA_EXP_FRAME_HEIGHT must be programmed to
//  an appropriate value if this fields is set to PAD0S.

#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_PAD_FRAME_PAD1S                   (1)    // //  Lines of all ones will be used to pad out frames
//  that are shorter than expected height.      
//  PPA_EXP_FRAME_HEIGHT must be programmed to
//  an appropriate value if this fields is set to PAD1S.

#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0_CSI_PPA_PAD_FRAME_NOPAD                   (2)


// Register CSI_PIXEL_STREAM_A_CONTROL1_0  // CSI Pixel Stream A Control 1
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL1_0                   (0x207)
// CSI Pixel Parser A Top Field Frame
//  This parameter specifies the frame number for 
//  top field detection for interlaced input video
//  stream. Top Field is indicated when each of the
//  least significant four bits of the frame number
//  that has a one in its mask bit matches the 
//  corresponding bit in this parameter. In other
//  words, Top Field is detected when the bitwise
//  AND of  
// ~(CSI_PPA_TOP_FIELD_FRAME ^ <frame number>) & CSI_PPA_TOP_FIELD_FRAME_MASK
//  is one. Frame Number is taken from the WC field
//  of the Frame Start short packet.
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL1_0_CSI_PPA_TOP_FIELD_FRAME                     3:0

// CSI Pixel Parser A Top Field Frame Mask
#define LWE24D_CSI_PIXEL_STREAM_A_CONTROL1_0_CSI_PPA_TOP_FIELD_FRAME_MASK                        7:4


// Register CSI_PIXEL_STREAM_A_WORD_COUNT_0  // CSI Pixel Stream A Word Count
#define LWE24D_CSI_PIXEL_STREAM_A_WORD_COUNT_0                 (0x208)
// CSI Pixel Parser A Word Count
//  This parameter specifies the number of
//  bytes per line/packet in the case where
//  Word Count field in packet header is not
//  used or where packet header is not sent.
//  This count does not includes the additional
//  2 bytes of CRC checksum if data CRC check
//  is enabled. 
//  When the input stream comes from a CSI camera
//  port, this parameter must be programmed when 
//  CSI_PPA_PAD_SHORT_LINE is set to either PAD0S
//  or PAD1S, no matter whether CSI_PPA_WORD_COUNT_SELECT
//  is set to REGISTER or HEADER.
//  When the input stream comes from the host path
//  or from the VIP path, and the data mode is
//  PAYLOAD_ONLY, this count must be programmed.
// Given a line width of N pixels, the programming 
//  value of this parameters is as follows
//  --------------------------------------
//  data format            value
//  --------------------------------------
//  YUV420_8               N bytes
//  YUV420_10              N/4*5 bytes
//  LEG_YUV420_8           N/2*3 bytes
//  YUV422_8               N*2 bytes
//  YUV422_10              N/2*5 bytes
//  RGB888                 N*3 bytes 
//  RGB666                 N/4*9 bytes                 
//  RGB565                 N*2 bytes
//  RGB555                 N*2 bytes 
//  RGB444                 N*2 bytes 
//  RAW6                   N/4*3 bytes
//  RAW7                   N/8*7 bytes
//  RAW8                   N bytes 
//  RAW10                  N/4*5 bytes
//  RAW12                  N/2*3 bytes
//  RAW14                  N/4*7 bytes
//  ---------------------------------------
#define LWE24D_CSI_PIXEL_STREAM_A_WORD_COUNT_0_CSI_PPA_WORD_COUNT                        15:0


// Register CSI_PIXEL_STREAM_A_GAP_0  // CSI Pixel Stream A Gap
#define LWE24D_CSI_PIXEL_STREAM_A_GAP_0                        (0x209)
// Minium number of viclk cycles from end of 
// previous line (Video_control = EL_DATA) to start
// of next line (Video_control = SL).
// This parameter is to ensure that minimum H-blank
// time requirement of VI/ISP is satisfied.
// This field takes effect only when the line gap 
// of the input stream is less than the specified
// value.
#define LWE24D_CSI_PIXEL_STREAM_A_GAP_0_PPA_LINE_MIN_GAP                 15:0

// Minium number of viclk cycles from end of 
// frame (Video_control = EF) to start of next
// frame (Video_control = SF).
// This parameter is to ensure that minimum V-blank
// time requirement of VI/ISP is satisfied.
// This field takes effect only when the frame gap 
// of the input stream is less than the specified
// value.
#define LWE24D_CSI_PIXEL_STREAM_A_GAP_0_PPA_FRAME_MIN_GAP                        31:16


// Register CSI_PIXEL_STREAM_PPA_COMMAND_0  // CSI Pixel Parser A Command
#define LWE24D_CSI_PIXEL_STREAM_PPA_COMMAND_0                  (0x20a)
// CSI Pixel Parser A Enable
//  This parameter controls CSI Pixel Parser A
//  to start or stop receiving data.       reset (disable immediately)
//  Enabling the pixel Parser does not enable 
//  the corresponding input source to receive 
//  data. If Pixel parser is enabled later than
//  the  corresponding input source, csi will keep
//  on rejecting incoming stream, till it encounters
//  a valid SF. 
#define LWE24D_CSI_PIXEL_STREAM_PPA_COMMAND_0_CSI_PPA_ENABLE                     1:0
#define LWE24D_CSI_PIXEL_STREAM_PPA_COMMAND_0_CSI_PPA_ENABLE_NOP                       (0)    // //       no operation

#define LWE24D_CSI_PIXEL_STREAM_PPA_COMMAND_0_CSI_PPA_ENABLE_ENABLE                    (1)    // //       enable at the next frame start as
//       specified by the CSI Start Marker

#define LWE24D_CSI_PIXEL_STREAM_PPA_COMMAND_0_CSI_PPA_ENABLE_DISABLE                   (2)    // //       disable after current frame end and before
//       next frame start.

#define LWE24D_CSI_PIXEL_STREAM_PPA_COMMAND_0_CSI_PPA_ENABLE_RST                       (3)

// CSI Pixel Parser A Single Shot Mode SW should Clear it along with disabling the 
// CSI_PPA_ENABLE, once a frame is captured
#define LWE24D_CSI_PIXEL_STREAM_PPA_COMMAND_0_CSI_PPA_SINGLE_SHOT                        2:2
#define LWE24D_CSI_PIXEL_STREAM_PPA_COMMAND_0_CSI_PPA_SINGLE_SHOT_DISABLE                      (0)
#define LWE24D_CSI_PIXEL_STREAM_PPA_COMMAND_0_CSI_PPA_SINGLE_SHOT_ENABLE                       (1)

// CSI Pixel Parser A VSYNC Start Marker  start of frame is indicated when VSYNC signal 
//  is received. When the input stream is from the
//  VIP path and the data mode is PACKET, then this
//  field may be programmed to VSYNC.
#define LWE24D_CSI_PIXEL_STREAM_PPA_COMMAND_0_CSI_PPA_VSYNC_START_MARKER                 4:4
#define LWE24D_CSI_PIXEL_STREAM_PPA_COMMAND_0_CSI_PPA_VSYNC_START_MARKER_FSPKT                 (0)    // //  Start of frame is indicated when a Frame
//  Start short packet is received with a frame
//  number whose least significant four bits are
//  greater than, or equal to, 
//  CSI_PPA_START_MARKER_FRAME_MIN and less than,
//  or equal to, CSI_PPA_START_MARKER_FRAME_MAX.
//  When the input stream is from a CSI port, or 
//  from the host path, or from the VIP path and 
//  the data mode is PAYLOAD_ONLY, then this field
//  may be programmed to FSPKT.

#define LWE24D_CSI_PIXEL_STREAM_PPA_COMMAND_0_CSI_PPA_VSYNC_START_MARKER_VSYNC                 (1)

// CSI Pixel Parser A Start Marker Minimum
// Start Frame is indicated when Max condition below
// is met and the least significant four bits of the
// frame number are greater than, or equal to, this value.
#define LWE24D_CSI_PIXEL_STREAM_PPA_COMMAND_0_CSI_PPA_START_MARKER_FRAME_MIN                     11:8

// CSI Pixel Parser A Start Marker Maximum
// Start Frame is indicated when Min condition above
// is met and the least significant four bits of the
// frame number are less than, or equal to, this value.
#define LWE24D_CSI_PIXEL_STREAM_PPA_COMMAND_0_CSI_PPA_START_MARKER_FRAME_MAX                     15:12





// reserved for additional Pixel Parser control registers
// in case it is needed in the future

// Register CSI_INPUT_STREAM_B_CONTROL_0  // CSI Input Stream B Control
#define LWE24D_CSI_INPUT_STREAM_B_CONTROL_0                    (0x20f)
// CSI-B Data Lane
//   0= 1 data lane
//   1= 2 data lanes (not supported on SC17 & SC25)
//   2= 3 data lanes (not supported on SC17 & SC25)
//   3= 4 data lanes (not supported on SC17 & SC25)
#define LWE24D_CSI_INPUT_STREAM_B_CONTROL_0_CSI_B_DATA_LANE                      1:0

// Enables skip packet threshold feature. Skip packet feature is enabled.
#define LWE24D_CSI_INPUT_STREAM_B_CONTROL_0_CSI_B_SKIP_PACKET_THRESHOLD_ENABLE                   4:4
#define LWE24D_CSI_INPUT_STREAM_B_CONTROL_0_CSI_B_SKIP_PACKET_THRESHOLD_ENABLE_DISABLE                 (0)    // // Skip packet feature is disabled.     

#define LWE24D_CSI_INPUT_STREAM_B_CONTROL_0_CSI_B_SKIP_PACKET_THRESHOLD_ENABLE_ENABLE                  (1)

// CSI-B Skip Packet Threshold
//  This value is compared against the internal
//  FIFO that buffer the input streams. A packet
//  will be skipped (discarded) if the pixel
//  stream processor is busy (probably due to
//  padding process of a short line) and the
//  number of entries in the internal FIFO
//  exceeds this threshold value. Note that
//  each entry in the internal FIFO buffer is
//  four bytes.
//  To turn off this feature, set the value
//  to its maximum value (all ones).
#define LWE24D_CSI_INPUT_STREAM_B_CONTROL_0_CSI_B_SKIP_PACKET_THRESHOLD                  22:16


// reserved for additional Input Stream control register
// in case it is needed in the future

// Register CSI_PIXEL_STREAM_B_CONTROL0_0  // CSI Pixel Stream A Control 0
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0                   (0x211)
// CSI Pixel Parser B Stream Source   Host
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_STREAM_SOURCE                       2:0
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_STREAM_SOURCE_CSI_A                       (0)    // //   CSI Interface A

#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_STREAM_SOURCE_CSI_B                       (1)    // //   CSI Interface B

#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_STREAM_SOURCE_VI_PORT                     (6)    // //   VI port

#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_STREAM_SOURCE_HOST                        (7)

// CSI Pixel Parser B Packet Header processing
//  This specifies whether packet header is
//  sent in the beginning of packet or not.      Packet header is sent.
//      This setting should be used if the
//      stream source is CSI Interface A or B.
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_PACKET_HEADER                       4:4
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_PACKET_HEADER_NOT_SENT                    (0)    // //      Packet header is not sent.
//      This setting should not be used if the
//      stream source is CSI Interface A or B.
//      Unless CSI-A, or CSI-B, is operating in a
//      stream capture debug mode.
//      In this case, CSI_PPB_DATA_TYPE specifies
//      the stream data format and the number
//      of bytes per line/packet is
//      specified by CSI_PPB_WORD_COUNT.
//      This implies that a packet footer
//      is also not sent.  In this case, no 
//      packet footer CRC check should be performed.

#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_PACKET_HEADER_SENT                        (1)

// CSI Pixel Parser B Data Identifier (DI) byte processing
//  This parameter is effective only if packet
//  header is sent as part of the stream.      Enabled  - Data Identifier byte in
//      packet header should be compared against
//      the CSI_PPB_DATA_TYPE and the
//      CSI_PPB_VIRTUAL_CHANNEL_ID.
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_DATA_IDENTIFIER                     5:5
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_DATA_IDENTIFIER_DISABLED                  (0)    // //      Disabled - Data Identifier byte in
//      packet header should be ignored
//      (not checked against CSI_PPB_DATA_TYPE
//      and against CSI_PPB_VIRTUAL_CHANNEL_ID).
//      In this case, CSI_PPB_DATA_TYPE specifies
//      the stream data format.

#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_DATA_IDENTIFIER_ENABLED                   (1)

// CSI Pixel Parser B Word Count Select
//  This parameter is effective only if packet
//  header is sent as part of the stream.      The number of bytes per line is to be
//      extracted from Word Count field in the
//      packet header. Note that if the serial
//      link is not error free, programming this
//      bit to HEADER may be dangerous because 
//      the word count information in the header
//      may be corrupted. 
//
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_WORD_COUNT_SELECT                   6:6
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_WORD_COUNT_SELECT_REGISTER                        (0)    // //      Word Count in packet header is ignored
//      and the number of bytes per line/packet
//      is specified by CSI_PPB_WORD_COUNT. Payload
//      CRC check will not be valid if the word
//      than the count in the packet header.
//      It is recommended to always program
//      this bit to REGISTER and always program
//      CSI_PPB_WORD_COUNT.

#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_WORD_COUNT_SELECT_HEADER                  (1)

// CSI Pixel Parser B Data CRC Check
//  This parameter specifies whether the last
//  2 bytes of packet should be treated as
//  CRC checksum and used to perform CRC check
//  on the payload data. Note that in case there
//  are 2 bytes of data CRC at the end of the
//  packet, the packet word count does not
//  include the CRC bytes.      Data CRC Check is enabled.
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_CRC_CHECK                   7:7
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_CRC_CHECK_DISABLE                 (0)    // //      Data CRC Check is disabled regardless
//      of whether there are CRC checksum at
//      the end of the packet.

#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_CRC_CHECK_ENABLE                  (1)

// CSI Pixel Parser B Data Type  This is CSI compatible data type as defined
//  in CSI specification. If the source stream
//  contains packet headers this value can be compared
//  to the CSI Data Type value in the 6 LSB of the
//  CSI Data Identifier (DI) byte. If the source stream
//  doesn't contain packet headers, or CSI_PPB_DATA_IDENTIFIER
//  is DISABLED, this value will be used to determine how
//  the stream will be colwerted to pixels.
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_DATA_TYPE                   13:8
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_DATA_TYPE_YUV420_8                        (24)
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_DATA_TYPE_YUV420_10                       (25)
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_DATA_TYPE_LEG_YUV420_8                    (26)
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_DATA_TYPE_YUV420CSPS_8                    (28)
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_DATA_TYPE_YUV420CSPS_10                   (29)
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_DATA_TYPE_YUV422_8                        (30)
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_DATA_TYPE_YUV422_10                       (31)
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_DATA_TYPE_RGB444                  (32)
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_DATA_TYPE_RGB555                  (33)
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_DATA_TYPE_RGB565                  (34)
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_DATA_TYPE_RGB666                  (35)
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_DATA_TYPE_RGB888                  (36)
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_DATA_TYPE_RAW6                    (40)
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_DATA_TYPE_RAW7                    (41)
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_DATA_TYPE_RAW8                    (42)
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_DATA_TYPE_RAW10                   (43)
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_DATA_TYPE_RAW12                   (44)
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_DATA_TYPE_RAW14                   (45)
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_DATA_TYPE_ARB_DT1                 (48)
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_DATA_TYPE_ARB_DT2                 (49)
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_DATA_TYPE_ARB_DT3                 (50)
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_DATA_TYPE_ARB_DT4                 (51)

// CSI Pixel Parser B Virtual Channel Identifier  
//  This is CSI compatible virtual channel
//  identifier as defined in CSI specification.
//  If the source stream contains packet headers
//  and CSI_PPB_DATA_IDENTIFIER is ENABLED this
//  value will be compared to the CSI Virtual
//  Channel Identifier value in the 2 MSB of the
//  CSI Data Identifier (DI) byte. This value will
//  be ignored if the source stream doesn't contain
//  packet headers, or CSI_PPB_DATA_IDENTIFIER is 
//  DISABLED, then this value will be ignored.
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_VIRTUAL_CHANNEL_ID                  15:14
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_VIRTUAL_CHANNEL_ID_ONE                    (0)
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_VIRTUAL_CHANNEL_ID_TWO                    (1)
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_VIRTUAL_CHANNEL_ID_THREE                  (2)
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_VIRTUAL_CHANNEL_ID_FOUR                   (3)

// CSI Pixel Parser B Output Format Options
//  This parameter specifies output data format.       Output for storing RAW data to memory through
//       ISP. Undefined LS color bits for RGB_666, 
//       RGB_565, RGB_555, and RGB_444, will be zeroed.
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_OUTPUT_FORMAT_OPTIONS                       19:16
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_OUTPUT_FORMAT_OPTIONS_ARBITRARY                   (0)    // //       Output as 8-bit arbitrary data stream
//       This may be used for compressed JPEG stream

#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_OUTPUT_FORMAT_OPTIONS_PIXEL                       (1)    // //       Output the normal 1 pixel/clock. Undefined 
//       LS color bits for RGB_666, RGB_565, RGB_555,
//       and RGB_444, will be zeroed.

#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_OUTPUT_FORMAT_OPTIONS_PIXEL_REP                   (2)    // //       Same as PIXEL except MS color bits, for RGB_666, 
//       RGB_565, RGB_555, and RGB_444, will be 
//       replicated to their undefined LS bits.

#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_OUTPUT_FORMAT_OPTIONS_STORE                       (3)

// CSI Pixel Parser B Embedded Data Options 
//  This specifies how to deal with embedded
//  data within the specified input stream
//  assuming that the CSI_PPB_DATA_TYPE is not
//  embedded data and assuming that embedded
//  data is not already processed by other
//  CSI pixel stream processor.       output embedded data as 8-bpp arbitrary
//       data stream.
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_EMBEDDED_DATA_OPTIONS                       21:20
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_EMBEDDED_DATA_OPTIONS_DISCARD                     (0)    // //       discard (throw away) embedded data

#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_EMBEDDED_DATA_OPTIONS_EMBEDDED                    (1)

// CSI Pixel Parser B Pad Short Line
//  This specifies how to deal with shorter than
//  expected line (the number of bytes received
//  is less than the specified word count)       short line is not padded (will output
//       less pixels than expected).
//       This option is not recommended and may
//       cause other modules that receives CSI
//       output stream to hang up.
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_PAD_SHORT_LINE                      25:24
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_PAD_SHORT_LINE_PAD0S                      (0)    // //       short line is padded by pixel of zeros
//       such that the expected number of output
//       pixels is correct. Due to the time
//       required to do the padding, subsequent
//       line packet maybe discarded and
//       therefore may cause a short frame
//       (total number of lines per frame is
//       less than expected).

#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_PAD_SHORT_LINE_PAD1S                      (1)    // //       short line is padded by pixel of ones
//       such that the expected number of output
//       pixels is correct. Due to the time
//       required to do the padding, subsequent
//       line packet maybe discarded and
//       therefore may cause a short frame
//       (total number of lines per frame is
//       less than expected).

#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_PAD_SHORT_LINE_NOPAD                      (2)

// CSI Pixel Parser B Packet Header Error Correction Enable
//  This parameter specifies whether single bit
//  errors in the packet header will be
//  automatically corrected, or not.    Single bit errors in the header will not
//    be corrected. Header ECC check will still
//    set header ECC status bits and the packet
//    will be processed by Pixel Parser B. DISABLE
//    should not be used when processing interleaved
//    streams (Same stream going to both PPA and PPB).
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_HEADER_EC_ENABLE                    27:27
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_HEADER_EC_ENABLE_ENABLE                   (0)    // //    Single bit errors in the header will be
//    automatically corrected.

#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_HEADER_EC_ENABLE_DISABLE                  (1)

// CSI Pixel Parser B Pad Frame
//  This specifies how to deal with frames that are
//  shorter (fewer lines) that expected. Short frames
//  are usually caused by line packets being dropped
//  because of packet errors. Expected frame height is
//  specified in PPB_EXP_FRAME_HEIGHT. To do padding the
//  number of input bytes in each lines payload.  Short frames will not be padded out.   
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_PAD_FRAME                   29:28
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_PAD_FRAME_PAD0S                   (0)    // //  Lines of all zeros will be used to pad out frames
//  that are shorter than expected height.   
//  PPB_EXP_FRAME_HEIGHT must be programmed to
//  an appropriate value if this fields is set to PAD0S.

#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_PAD_FRAME_PAD1S                   (1)    // //  Lines of all ones will be used to pad out frames
//  that are shorter than expected height.      
//  PPB_EXP_FRAME_HEIGHT must be programmed to
//  an appropriate value if this fields is set to PAD1S.

#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0_CSI_PPB_PAD_FRAME_NOPAD                   (2)

// Register CSI_PIXEL_STREAM_B_CONTROL1_0  // CSI Pixel Stream B Control 1
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL1_0                   (0x212)
// CSI Pixel Parser B Top Field Frame
//  This parameter specifies the frame number for 
//  top field detection for interlaced input video
//  stream. Top Field is indicated when each of the
//  least significant four bits of the frame number
//  that has a one in its mask bit matches the 
//  corresponding bit in this parameter. In other
//  words, Top Field is detected when the bitwise
//  AND of  
// ~(CSI_PPB_TOP_FIELD_FRAME ^ <frame number>) & CSI_PPB_TOP_FIELD_FRAME_MASK
//  is one. Frame Number is taken from the WC field
//  of the Frame Start short packet.
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL1_0_CSI_PPB_TOP_FIELD_FRAME                     3:0

// CSI Pixel Parser B Top Field Frame Mask
#define LWE24D_CSI_PIXEL_STREAM_B_CONTROL1_0_CSI_PPB_TOP_FIELD_FRAME_MASK                        7:4

// Register CSI_PIXEL_STREAM_B_WORD_COUNT_0  // CSI Pixel Stream A Word Count
#define LWE24D_CSI_PIXEL_STREAM_B_WORD_COUNT_0                 (0x213)
// CSI Pixel Parser B Word Count
//  This parameter specifies the number of
//  bytes per line/packet in the case where
//  Word Count field in packet header is not
//  used or where packet header is not sent.
//  This count does not includes the additional
//  2 bytes of CRC checksum if data CRC check
//  is enabled.
//  When the input stream comes from a CSI camera
//  port, this parameter must be programmed when 
//  CSI_PPB_PAD_SHORT_LINE is set to either PAD0S
//  or PAD1S, no matter whether CSI_PPB_WORD_COUNT_SELECT
//  is set to REGISTER or HEADER.
//  When the input stream comes from the host path
//  or from the VIP path, and the data mode is
//  PAYLOAD_ONLY, this count must be programmed.
// Given a line width of N pixels, the programming 
//  value of this parameters is as follows
//  --------------------------------------
//  data format            value
//  --------------------------------------
//  YUV420_8               N bytes
//  YUV420_10              N/4*5 bytes
//  LEG_YUV420_8           N/2*3 bytes
//  YUV422_8               N*2 bytes
//  YUV422_10              N/2*5 bytes
//  RGB888                 N*3 bytes 
//  RGB666                 N/4*9 bytes                 
//  RGB565                 N*2 bytes
//  RGB555                 N*2 bytes 
//  RGB444                 N*2 bytes 
//  RAW6                   N/4*3 bytes
//  RAW7                   N/8*7 bytes
//  RAW8                   N bytes 
//  RAW10                  N/4*5 bytes
//  RAW12                  N/2*3 bytes
//  RAW14                  N/4*7 bytes
//  ---------------------------------------
#define LWE24D_CSI_PIXEL_STREAM_B_WORD_COUNT_0_CSI_PPB_WORD_COUNT                        15:0

// Register CSI_PIXEL_STREAM_B_GAP_0  // CSI Pixel Stream B Gap
#define LWE24D_CSI_PIXEL_STREAM_B_GAP_0                        (0x214)
// Minium number of viclk cycles from end of 
// previous line (Video_control = EL_DATA) to start
// of next line (Video_control = SL).
// This parameter is to ensure that minimum H-blank
// time requirement of VI/ISP is satisfied.
// This field takes effect only when the line gap 
// of the input stream is less than the specified
// value.
#define LWE24D_CSI_PIXEL_STREAM_B_GAP_0_PPB_LINE_MIN_GAP                 15:0

// Minium number of viclk cycles from end of 
// frame (Video_control = EF) to start of next
// frame (Video_control = SF).
// This parameter is to ensure that minimum V-blank
// time requirement of VI/ISP is satisfied.
// This field takes effect only when the frame gap 
// of the input stream is less than the specified
// value.
#define LWE24D_CSI_PIXEL_STREAM_B_GAP_0_PPB_FRAME_MIN_GAP                        31:16

// Register CSI_PIXEL_STREAM_PPB_COMMAND_0  // CSI Pixel Parser B Command
#define LWE24D_CSI_PIXEL_STREAM_PPB_COMMAND_0                  (0x215)
// CSI Pixel Parser B Enable
//  This parameter controls CSI Pixel Parser B
//  to start or stop receiving data.       reset (disable immediately)
//  Enabling the pixel Parser does not enable 
//  the corresponding input source to receive 
//  data. If Pixel parser is enabled later than
//  the  corresponding input source, csi will keep
//  on rejecting incoming stream, till it encounters
//  a valid SF.
#define LWE24D_CSI_PIXEL_STREAM_PPB_COMMAND_0_CSI_PPB_ENABLE                     1:0
#define LWE24D_CSI_PIXEL_STREAM_PPB_COMMAND_0_CSI_PPB_ENABLE_NOP                       (0)    // //       no operation

#define LWE24D_CSI_PIXEL_STREAM_PPB_COMMAND_0_CSI_PPB_ENABLE_ENABLE                    (1)    // //       enable at the next frame start as
//       specified by the CSI Start Marker

#define LWE24D_CSI_PIXEL_STREAM_PPB_COMMAND_0_CSI_PPB_ENABLE_DISABLE                   (2)    // //       disable after current frame end and before
//       next frame start.

#define LWE24D_CSI_PIXEL_STREAM_PPB_COMMAND_0_CSI_PPB_ENABLE_RST                       (3)

// CSI Pixel Parser B Single Shot Mode SW should Clear it alongwith disabling the 
// CSI_PPB_ENABLE, once a frame is captured
#define LWE24D_CSI_PIXEL_STREAM_PPB_COMMAND_0_CSI_PPB_SINGLE_SHOT                        2:2
#define LWE24D_CSI_PIXEL_STREAM_PPB_COMMAND_0_CSI_PPB_SINGLE_SHOT_DISABLE                      (0)
#define LWE24D_CSI_PIXEL_STREAM_PPB_COMMAND_0_CSI_PPB_SINGLE_SHOT_ENABLE                       (1)

// CSI Pixel Parser B VSYNC Start Marker  Start of frame is indicated when VSYNC signal 
//  is received. When the input stream is from the
//  VIP path and the data mode is PACKET, then this
//  field may be programmed to VSYNC.
#define LWE24D_CSI_PIXEL_STREAM_PPB_COMMAND_0_CSI_PPB_VSYNC_START_MARKER                 4:4
#define LWE24D_CSI_PIXEL_STREAM_PPB_COMMAND_0_CSI_PPB_VSYNC_START_MARKER_FSPKT                 (0)    // //      Start of frame is indicated when a Frame
//    Start short packet is received with a frame
//    number who's least significant four bits are
//    greater than, or equal to, 
//    CSI_PPB_START_MARKER_FRAME_MIN and less than,
//    or equal to, CSI_PPB_START_MARKER_FRAME_MAX.
//  When the input stream is from a CSI port, or 
//  from the host path, or from the VIP path and 
//  the data mode is PAYLOAD_ONLY, then this field
//  may be programmed to FSPKT.

#define LWE24D_CSI_PIXEL_STREAM_PPB_COMMAND_0_CSI_PPB_VSYNC_START_MARKER_VSYNC                 (1)

// CSI Pixel Parser B Start Marker Minimum
// Start Frame is indicated when Max condition below
// is met and the least significant four bits of the
// frame number are greater than, or equal to, this value.
#define LWE24D_CSI_PIXEL_STREAM_PPB_COMMAND_0_CSI_PPB_START_MARKER_FRAME_MIN                     11:8

// CSI Pixel Parser B Start Marker Maximum
// Start Frame is indicated when Min condition above
// is met and the least significant four bits of the
// frame number are less than, or equal to, this value.
#define LWE24D_CSI_PIXEL_STREAM_PPB_COMMAND_0_CSI_PPB_START_MARKER_FRAME_MAX                     15:12

// reserved for additional Pixel Parser control registers
// in case it is needed in the future

// Register CSI_PHY_CIL_COMMAND_0  // CSI Phy and CIL Command
#define LWE24D_CSI_PHY_CIL_COMMAND_0                   (0x21a)
// CSI A Phy and CIL Enable
//  This parameter controls CSI A Phy and CIL
//  receiver to start or stop receiving data.    disable (reset)
#define LWE24D_CSI_PHY_CIL_COMMAND_0_CSI_A_PHY_CIL_ENABLE                        1:0
#define LWE24D_CSI_PHY_CIL_COMMAND_0_CSI_A_PHY_CIL_ENABLE_NOP                  (0)    // //    no operation

#define LWE24D_CSI_PHY_CIL_COMMAND_0_CSI_A_PHY_CIL_ENABLE_ENABLE                       (1)    // //    enable

#define LWE24D_CSI_PHY_CIL_COMMAND_0_CSI_A_PHY_CIL_ENABLE_DISABLE                      (2)

// CSI B Phy and CIL Enable
//  This parameter controls CSI B Phy and CIL
//  receiver to start or stop receiving data.    disable (reset)
#define LWE24D_CSI_PHY_CIL_COMMAND_0_CSI_B_PHY_CIL_ENABLE                        17:16
#define LWE24D_CSI_PHY_CIL_COMMAND_0_CSI_B_PHY_CIL_ENABLE_NOP                  (0)    // //    no operation

#define LWE24D_CSI_PHY_CIL_COMMAND_0_CSI_B_PHY_CIL_ENABLE_ENABLE                       (1)    // //    enable

#define LWE24D_CSI_PHY_CIL_COMMAND_0_CSI_B_PHY_CIL_ENABLE_DISABLE                      (2)

// Register CSI_PHY_CILA_CONTROL0_0  // CSI-A Phy and CIL Control
#define LWE24D_CSI_PHY_CILA_CONTROL0_0                 (0x21b)
// When moving from LP mode to High Speed (LP11->LP01->LP00),
// this setting determines how many csicil clock cycles (72 MHz
// lp clock cycles) to wait, after LP00, 
// before starting to look at the data.
#define LWE24D_CSI_PHY_CILA_CONTROL0_0_CILA_THS_SETTLE                   3:0

// The LP signals are sampled using csi_cil_clk.
// Normally this happens on 2 clock edges assuming
// the clock is running at least 50 Mhz.  If the
// clock needs to run slower, then this bit can be
// SET so that the sampling takes place on a single
// edge (clock rate is 25 Mhz min).  This sampling
// may not be as reliable so setting this bit is
// not recommended.
#define LWE24D_CSI_PHY_CILA_CONTROL0_0_CILA_SINGLE_SAMPLE                        4:4

// The LP signals should sequence through LP11->LP01->LP00 state,
// to indicate to CLOCK CIL about the mode switching to HS Rx mode.
// In case Camera is enabled earlier than CIL , it is highly likely
// that camera sends this control sequence sooner than cil can detect it.
// Enabling this bit allows the CLOCK CIL to overlook the LP control sequence
// and step in HS Rx mode directly looking at LP00 only.
#define LWE24D_CSI_PHY_CILA_CONTROL0_0_CILA_BYPASS_LP_SEQ                        5:5

// Register CSI_PHY_CILB_CONTROL0_0  // CSI-B Phy and CIL Control
#define LWE24D_CSI_PHY_CILB_CONTROL0_0                 (0x21c)
// When moving from LP mode to High Speed (LP11->LP01->LP00),
// this setting determines how many  csicil clock cycles (72 MHz
// lp clock cycles) to wait, after LP00,
// before starting to look at the data.
#define LWE24D_CSI_PHY_CILB_CONTROL0_0_CILB_THS_SETTLE                   3:0

// see CILA_SINGLE_SAMPLE above
#define LWE24D_CSI_PHY_CILB_CONTROL0_0_CILB_SINGLE_SAMPLE                        4:4

// see CILA_BYPASS_LP_SEQ above
#define LWE24D_CSI_PHY_CILB_CONTROL0_0_CILB_BYPASS_LP_SEQ                        5:5

// reserved for additional Input Stream control register
// in case it is needed in the future

// Register CSI_CSI_PIXEL_PARSER_STATUS_0  // Pixel Parser Status
// These status bits are cleared to
// zero when its bit position is written with one. For
// example write 0x2 to CSI_PIXEL_PARSER_STATUS will 
// clear only PPA_ILL_WD_CNT.
#define LWE24D_CSI_CSI_PIXEL_PARSER_STATUS_0                   (0x21e)
// Header Error Corrected, Set when a packet that was
// processed by PPA has a single bit header error. This error
// will be detected by the headers ECC, and corrected by
// it if header error correction is enabled 
// (CSI_A_HEADER_EC_ENABLE = 0). This flag will be set and 
// the packet will be processed even if the error is not
// corrected.
#define LWE24D_CSI_CSI_PIXEL_PARSER_STATUS_0_PPA_HDR_ERR_COR                     0:0

// Illegal Word Count, set when a line with a word count that
// doesn't generate an integer number of pixels (Unused bytes
// at the end of payload) is processed by PPA.
#define LWE24D_CSI_CSI_PIXEL_PARSER_STATUS_0_PPA_ILL_WD_CNT                      1:1

// Short Line Processed, Set when a line with a payload that
// is shorter than its packet header word count is processed
// by PPA.
#define LWE24D_CSI_CSI_PIXEL_PARSER_STATUS_0_PPA_SL_PROCESSED                    2:2

// Short Line Packet Dropped, set when a in coming packet
// gets dropped because the input FIFO level reaches
// CSI_A_SKIP_PACKET_THRESHOLD when padding a short line.
#define LWE24D_CSI_CSI_PIXEL_PARSER_STATUS_0_PPA_SL_PKT_DROPPED                  3:3

// PayLoad CRC Error, Set when a packet that was processed by
// PPA had a payload CRC error.
#define LWE24D_CSI_CSI_PIXEL_PARSER_STATUS_0_PPA_PL_CRC_ERR                      4:4

// FIFO Overflow, set when the fifo that is feeding packets
// to PPA overflows.
#define LWE24D_CSI_CSI_PIXEL_PARSER_STATUS_0_PPA_FIFO_OVRF                       5:5

// Stream Error, set when the control output of PPA doesn't
// follow the correct sequence. The correct sequence for CSI
// is: SF -> (SL_DATA or EF), SL_DATA -> (DATA or EL_DATA),
// DATA -> EL_DATA, EL_DATA -> (SL_DATA or EF), EF -> SF.
// Stream Errors can be caused by receiving a corrupted
// stream, or a CSI RTL bug.
#define LWE24D_CSI_CSI_PIXEL_PARSER_STATUS_0_PPA_STMERR                  6:6

// Set when CSI-PPA receives a short frame. This bit gets
// set even if CSI_PPA_PAD_FRAME specifies that short frames
// are to be padded to the correct line length.
#define LWE24D_CSI_CSI_PIXEL_PARSER_STATUS_0_PPA_SHORT_FRAME                     7:7

// Set when CSI-PPA receives a SF when it is expecting an EF.
// This happens when EF of the frame gets corrupted before arriving CSI.
// CSI-PPA will insert a fake EF and the drop the current 
// frame with Correct SF.
#define LWE24D_CSI_CSI_PIXEL_PARSER_STATUS_0_PPA_EXTRA_SF                        8:8

// Set when CSI-PPA receives a request to output a line
// that is not in the active part of the frame output. That
// is after EF and before SF, or before start marker is found.
// The interframe line will not be outputted by the Pixel 
// Parser.
#define LWE24D_CSI_CSI_PIXEL_PARSER_STATUS_0_PPA_INTERFRAME_LINE                 9:9

// PPA Spare Status bit. This bit will get set when Pixel Parser
// A has a line timeout. Line timeout needs to be enabled by setting
// PPA_ENABLE_LINE_TIMEOUT and programming PPA_MAX_CLOCKS for
// the MAX clocks between lines.
#define LWE24D_CSI_CSI_PIXEL_PARSER_STATUS_0_PPA_SPARE_STATUS_1                  10:10

// PPA Spare Status bit.
#define LWE24D_CSI_CSI_PIXEL_PARSER_STATUS_0_PPA_SPARE_STATUS_2                  11:11

// Uncorrectable Header Error, Set when header parser A
// parses a header with a multi bit error. This error will
// be detected by the headers ECC, but can't be corrected.
// The packet will be discarded.
#define LWE24D_CSI_CSI_PIXEL_PARSER_STATUS_0_HPA_UNC_HDR_ERR                     14:14

// Uncorrectable Header Error, Set when header parser B
// parses a header with a multi bit error. This error will
// be detected by the headers ECC, but can't be corrected.
// The packet will be discarded.a multi bit error. This error will
// be detected by the headers ECC, but can't be corrected.
// The packet will be discarded.
#define LWE24D_CSI_CSI_PIXEL_PARSER_STATUS_0_HPB_UNC_HDR_ERR                     15:15

// Header Error Corrected, set when a packet that was
// processed by PPB has a single bit header error. This error
// will be detected by the headers ECC, and corrected by
// it if header error correction is enabled 
// (CSI_B_HEADER_EC_ENABLE = 0). This flag will be set and 
// the packet will be processed even if the error is not
// corrected.
#define LWE24D_CSI_CSI_PIXEL_PARSER_STATUS_0_PPB_HDR_ERR_COR                     16:16

// Illegal Word Count, set when a line with a word count that
// doesn't generate an integer number of pixels (Unused bytes
// at the end of payload) is processed by PPB.
#define LWE24D_CSI_CSI_PIXEL_PARSER_STATUS_0_PPB_ILL_WD_CNT                      17:17

// Short Line Processed, Set when a line with a payload that
// is shorter than its packet header word count is processed
// by PPB.
#define LWE24D_CSI_CSI_PIXEL_PARSER_STATUS_0_PPB_SL_PROCESSED                    18:18

// Short Line Packet Dropped, set when a in coming packet
// gets dropped because the input FIFO level reaches
// CSI_B_SKIP_PACKET_THRESHOLD when padding a short line.
#define LWE24D_CSI_CSI_PIXEL_PARSER_STATUS_0_PPB_SL_PKT_DROPPED                  19:19

// PayLoad CRC Error, Set when a packet that was processed
// by PPB had a payload CRC error.
#define LWE24D_CSI_CSI_PIXEL_PARSER_STATUS_0_PPB_PL_CRC_ERR                      20:20

// FIFO Overflow, set when the fifo that is feeding packets
// to PPB overflows.
#define LWE24D_CSI_CSI_PIXEL_PARSER_STATUS_0_PPB_FIFO_OVRF                       21:21

// Stream Error, set when the control output of PPB doesn't
// follow the correct sequence. The correct sequence for CSI
// is: SF -> (SL_DATA or EF), SL_DATA -> (DATA or EL_DATA),
// DATA -> EL_DATA, EL_DATA -> (SL_DATA or EF), EF -> SF.
// Stream Errors can be caused by receiving a corrupted
// stream, or a CSI RTL bug.
#define LWE24D_CSI_CSI_PIXEL_PARSER_STATUS_0_PPB_STMERR                  22:22

// Set when CSI-PPB receives a short frame. This bit gets
// set even if CSI_PPB_PAD_FRAME specifies that short frames
// are to be padded to the correct line length.
#define LWE24D_CSI_CSI_PIXEL_PARSER_STATUS_0_PPB_SHORT_FRAME                     23:23

// Set when CSI-PPB receives a SF when it is expecting an EF. 
// This happens when EF of the frame gets corrupted before arriving CSI.
// CSI-PPB will insert a fake EF and the drop the current 
// frame with Correct SF.
#define LWE24D_CSI_CSI_PIXEL_PARSER_STATUS_0_PPB_EXTRA_SF                        24:24

// Set when CSI-PPB receives a request to output a line
// that is not in the active part of the frame output. That
// is after EF and before SF, or before start marker is found.
// The interframe line will not be outputted by the Pixel 
// Parser.
#define LWE24D_CSI_CSI_PIXEL_PARSER_STATUS_0_PPB_INTERFRAME_LINE                 25:25

// PPB Spare Status bit. This bit will get set when Pixel Parser
// B has a line timeout. Line timeout needs to be enabled by setting
// PPB_ENABLE_LINE_TIMEOUT and programming PPB_MAX_CLOCKS for
// the MAX clocks between lines.
#define LWE24D_CSI_CSI_PIXEL_PARSER_STATUS_0_PPB_SPARE_STATUS_1                  26:26

// PPB Spare Status bit.
#define LWE24D_CSI_CSI_PIXEL_PARSER_STATUS_0_PPB_SPARE_STATUS_2                  27:27

// Uncorrectable Header Error, Set when the VI port header
// parser parses a header with a multi bit error. This error
// will be detected by the headers ECC, but can't be corrected.
// The packet will be discarded.
#define LWE24D_CSI_CSI_PIXEL_PARSER_STATUS_0_HPV_UNC_HDR_ERR                     30:30

// Uncorrectable Header Error, Set when the Host port header
// parser parses a header with a multi bit error. This error
// will be detected by the headers ECC, but can't be corrected.
// The packet will be discarded.
#define LWE24D_CSI_CSI_PIXEL_PARSER_STATUS_0_HPH_UNC_HDR_ERR                     31:31


// Register CSI_CSI_CIL_STATUS_0  // CSI Control and Interface Logic Status
// These status bits are cleared to
// zero when its bit position is written with one. For
// example write 0x2 to CSI_CIL_STATUS will clear only
// CILA_SOT_MB_ERR.
#define LWE24D_CSI_CSI_CIL_STATUS_0                    (0x21f)
// Start of Transmission Single Bit Error, set when CIL-A 
// detects a single bit error in one of the 
// packets Start of Transmission bytes. The packet will be
// sent to the CSI-A for processing.
#define LWE24D_CSI_CSI_CIL_STATUS_0_CILA_SOT_SB_ERR                      0:0

// Start of Transmission Multi Bit Error, set when CIL-A
// detects a multi bit start of transmission byte error in
// one of the packets SOT bytes. The packet will be discarded.
#define LWE24D_CSI_CSI_CIL_STATUS_0_CILA_SOT_MB_ERR                      1:1

// Sync Escape Error, set when CIL-A detects that the wrong 
// (non-multiple of 8) number of bits have been received for
// an Escape Command, or Data Byte.
#define LWE24D_CSI_CSI_CIL_STATUS_0_CILA_SYNC_ESC_ERR                    2:2

// Escape Mode Entry Error, set when CIL-A detects an escape
// mode entry error. The Escape mode command byte will not be
// received.
#define LWE24D_CSI_CSI_CIL_STATUS_0_CILA_ESC_ENTRY_ERR                   3:3

// Control Error, set when CIL-A detects LP state 01 or 10
// followed by a stop state (LP11) instead of transitioning
// into the Escape mode or Turn Around mode (LP00).
#define LWE24D_CSI_CSI_CIL_STATUS_0_CILA_CTRL_ERR                        4:4

// Escape Mode Command Received, set when CIL-A receives an
// Escape Mode Command byte. The Command Byte can be read 
// from bits 7-0 of ESCAPE_MODE_COMMAND.
#define LWE24D_CSI_CSI_CIL_STATUS_0_CILA_ESC_CMD_REC                     5:5

// Escape Mode Data Received, set when CIL-A receives an
// Escape Mode Data byte. The Data Byte can be read 
// from bits 7-0 of ESCAPE_MODE_DATA. This status bit will
// will also be cleared when CILA_ESC_CMD_REC is set.
#define LWE24D_CSI_CSI_CIL_STATUS_0_CILA_ESC_DATA_REC                    6:6

// CILA Spare Status bit.
#define LWE24D_CSI_CSI_CIL_STATUS_0_CILA_SPARE_STATUS_1                  7:7

// CILA Spare Status bit.
#define LWE24D_CSI_CSI_CIL_STATUS_0_CILA_SPARE_STATUS_2                  8:8

// MIPI Auto Calibrate done, set when the auto calibrate 
// sequence for MIPI pad bricks is done.
#define LWE24D_CSI_CSI_CIL_STATUS_0_MIPI_AUTO_CAL_DONE                   15:15

// Start of Transmission Single Bit Error, set when CIL-B
// detects a single bit error in one of the packets start
// of transmission bytes. The packet will be sent to CSI-B
// for processing.
#define LWE24D_CSI_CSI_CIL_STATUS_0_CILB_SOT_SB_ERR                      16:16

// Start of Transmission Multi Bit Error, set when CIL-B
// detects a multi bit start of transmission byte error in
// one of the packets SOT bytes. The packet will be discarded.
#define LWE24D_CSI_CSI_CIL_STATUS_0_CILB_SOT_MB_ERR                      17:17

// Sync Escape Error, set when CIL-B detects that the wrong 
// (non-multiple of 8) number of bits have been received for
// an Escape Command, or Data Byte.
#define LWE24D_CSI_CSI_CIL_STATUS_0_CILB_SYNC_ESC_ERR                    18:18

// Escape Mode Entry Error, set when CIL-B detects an Escape
// Mode Entry Error. The Escape mode command byte will not be
// received.
#define LWE24D_CSI_CSI_CIL_STATUS_0_CILB_ESC_ENTRY_ERR                   19:19

// Control Error, set when CIL-B detects LP state 01 or 10
// followed by a stop state (LP11) instead of transitioning
// into the Escape mode or Turn Around mode (LP00)..
#define LWE24D_CSI_CSI_CIL_STATUS_0_CILB_CTRL_ERR                        20:20

// Escape Mode Command Received, set when CIL-B receives an
// Escape Mode Command byte. The Command Byte can be read 
// from bits 23-16 of ESCAPE_MODE_COMMAND.
#define LWE24D_CSI_CSI_CIL_STATUS_0_CILB_ESC_CMD_REC                     21:21

// Escape Mode Data Received, set when CIL-B receives an
// Escape Mode Data byte. The Data Byte can be read 
// from bits 23-16 of ESCAPE_MODE_DATA. This status bit will
// will also be cleared when CILB_ESC_CMD_REC is set.
#define LWE24D_CSI_CSI_CIL_STATUS_0_CILB_ESC_DATA_REC                    22:22

// CILB Spare Status bit.
#define LWE24D_CSI_CSI_CIL_STATUS_0_CILB_SPARE_STATUS_1                  23:23

// CILB Spare Status bit.
#define LWE24D_CSI_CSI_CIL_STATUS_0_CILB_SPARE_STATUS_2                  24:24


// Register CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0  // CSI Pixel Parser Interrupt Mask
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0                   (0x220)
// Interrupt Mask for PPA_HDR_ERR_COR. Generate an interrupt when PPA_HDR_ERR_COR
// is set.
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_HDR_ERR_COR_INT_MASK                    0:0
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_HDR_ERR_COR_INT_MASK_DISABLED                 (0)    // // Don't generate an interrupt when PPA_HDR_ERR_COR
// is set.

#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_HDR_ERR_COR_INT_MASK_ENABLED                  (1)

// Interrupt Mask for PPA_ILL_WD_CNT. Generate an interrupt when PPA_ILL_WD_CNT
// is set.
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_ILL_WD_CNT_INT_MASK                     1:1
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_ILL_WD_CNT_INT_MASK_DISABLED                  (0)    // // Don't generate an interrupt when PPA_ILL_WD_CNT
// is set.

#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_ILL_WD_CNT_INT_MASK_ENABLED                   (1)

// Interrupt Mask for PPA_SL_PROCESSED. Generate an interrupt when PPA_SL_PROCESSED
// is set.
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_SL_PROCESSED_INT_MASK                   2:2
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_SL_PROCESSED_INT_MASK_DISABLED                        (0)    // // Don't generate an interrupt when PPA_SL_PROCESSED
// is set.

#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_SL_PROCESSED_INT_MASK_ENABLED                 (1)

// Interrupt Mask for PPA_SL_PKT_DROPPED. Generate an interrupt when PPA_SL_PKT_DROPPED
// is set.
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_SL_PKT_DROPPED_INT_MASK                 3:3
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_SL_PKT_DROPPED_INT_MASK_DISABLED                      (0)    // // Don't generate an interrupt when PPA_SL_PKT_DROPPED
// is set.

#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_SL_PKT_DROPPED_INT_MASK_ENABLED                       (1)

// Interrupt Mask for PPA_PL_CRC_ERR. Generate an interrupt when PPA_PL_CRC_ERR
// is set.
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_PL_CRC_ERR_INT_MASK                     4:4
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_PL_CRC_ERR_INT_MASK_DISABLED                  (0)    // // Don't generate an interrupt when PPA_PL_CRC_ERR
// is set.

#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_PL_CRC_ERR_INT_MASK_ENABLED                   (1)

// Interrupt Mask for PPA_FIFO_OVRF. Generate an interrupt when PPA_FIFO_OVRF
// is set.
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_FIFO_OVRF_INT_MASK                      5:5
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_FIFO_OVRF_INT_MASK_DISABLED                   (0)    // // Don't generate an interrupt when PPA_FIFO_OVRF
// is set.

#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_FIFO_OVRF_INT_MASK_ENABLED                    (1)

// Interrupt Mask for PPA_STMERR. Generate an interrupt when PPA_STMERR
// is set.
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_STMERR_INT_MASK                 6:6
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_STMERR_INT_MASK_DISABLED                      (0)    // // Don't generate an interrupt when PPA_STMERR
// is set.

#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_STMERR_INT_MASK_ENABLED                       (1)

// Interrupt Mask for PPA_SHORT_FRAME. Generate an interrupt when PPA_SHORT_FRAME
// is set.
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_SHORT_FRAME_INT_MASK                    7:7
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_SHORT_FRAME_INT_MASK_DISABLED                 (0)    // // Don't generate an interrupt when PPA_SHORT_FRAME
// is set.

#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_SHORT_FRAME_INT_MASK_ENABLED                  (1)

// Interrupt Mask for PPA_EXTRA_SF. Generate an interrupt when PPA_EXTRA_SF
// is set.
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_EXTRA_SF_INT_MASK                       8:8
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_EXTRA_SF_INT_MASK_DISABLED                    (0)    // // Don't generate an interrupt when PPA_EXTRA_SF
// is set.

#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_EXTRA_SF_INT_MASK_ENABLED                     (1)

// Interrupt Mask for PPA_INTERFRAME_LINE. Generate an interrupt when PPA_INTERFRAME_LINE
// is set.
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_INTERFRAME_LINE_INT_MASK                        9:9
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_INTERFRAME_LINE_INT_MASK_DISABLED                     (0)    // // Don't generate an interrupt when PPA_INTERFRAME_LINE
// is set.

#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_INTERFRAME_LINE_INT_MASK_ENABLED                      (1)

// Interrupt Mask for PPA_SPARE_STATUS_1. Generate an interrupt when PPA_SPARE_STATUS_1
// is set.
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_SPARE_STATUS_1_INT_MASK                 10:10
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_SPARE_STATUS_1_INT_MASK_DISABLED                      (0)    // // Don't generate an interrupt when PPA_SPARE_STATUS_1
// is set.

#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_SPARE_STATUS_1_INT_MASK_ENABLED                       (1)

// Interrupt Mask for PPA_SPARE_STATUS_2. Generate an interrupt when PPA_SPARE_STATUS_2
// is set.
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_SPARE_STATUS_2_INT_MASK                 11:11
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_SPARE_STATUS_2_INT_MASK_DISABLED                      (0)    // // Don't generate an interrupt when PPA_SPARE_STATUS_2
// is set.

#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPA_SPARE_STATUS_2_INT_MASK_ENABLED                       (1)

// Interrupt Mask for HPA_UNC_HDR_ERR. Generate an interrupt when HPA_UNC_HDR_ERR
// is set.
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_HPA_UNC_HDR_ERR_INT_MASK                    14:14
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_HPA_UNC_HDR_ERR_INT_MASK_DISABLED                 (0)    // // Don't generate an interrupt when HPA_UNC_HDR_ERR
// is set.

#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_HPA_UNC_HDR_ERR_INT_MASK_ENABLED                  (1)

// Interrupt Mask for HPB_UNC_HDR_ERR. Generate an interrupt when HPB_UNC_HDR_ERR
// is set.
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_HPB_UNC_HDR_ERR_INT_MASK                    15:15
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_HPB_UNC_HDR_ERR_INT_MASK_DISABLED                 (0)    // // Don't generate an interrupt when HPB_UNC_HDR_ERR
// is set.

#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_HPB_UNC_HDR_ERR_INT_MASK_ENABLED                  (1)

// Interrupt Mask for PPB_HDR_ERR_COR. Generate an interrupt when PPB_HDR_ERR_COR
// is set.
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_HDR_ERR_COR_INT_MASK                    16:16
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_HDR_ERR_COR_INT_MASK_DISABLED                 (0)    // // Don't generate an interrupt when PPB_HDR_ERR_COR
// is set.

#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_HDR_ERR_COR_INT_MASK_ENABLED                  (1)

// Interrupt Mask for PPB_ILL_WD_CNT. Generate an interrupt when PPB_ILL_WD_CNT
// is set.
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_ILL_WD_CNT_INT_MASK                     17:17
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_ILL_WD_CNT_INT_MASK_DISABLED                  (0)    // // Don't generate an interrupt when PPB_ILL_WD_CNT
// is set.

#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_ILL_WD_CNT_INT_MASK_ENABLED                   (1)

// Interrupt Mask for PPB_SL_PROCESSED. Generate an interrupt when PPB_SL_PROCESSED
// is set.
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_SL_PROCESSED_INT_MASK                   18:18
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_SL_PROCESSED_INT_MASK_DISABLED                        (0)    // // Don't generate an interrupt when PPB_SL_PROCESSED
// is set.

#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_SL_PROCESSED_INT_MASK_ENABLED                 (1)

// Interrupt Mask for PPB_SL_PKT_DROPPED. Generate an interrupt when PPB_SL_PKT_DROPPED
// is set.
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_SL_PKT_DROPPED_INT_MASK                 19:19
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_SL_PKT_DROPPED_INT_MASK_DISABLED                      (0)    // // Don't generate an interrupt when PPB_SL_PKT_DROPPED
// is set.

#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_SL_PKT_DROPPED_INT_MASK_ENABLED                       (1)

// Interrupt Mask for PPB_PL_CRC_ERR. Generate an interrupt when PPB_PL_CRC_ERR
// is set.
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_PL_CRC_ERR_INT_MASK                     20:20
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_PL_CRC_ERR_INT_MASK_DISABLED                  (0)    // // Don't generate an interrupt when PPB_PL_CRC_ERR
// is set.

#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_PL_CRC_ERR_INT_MASK_ENABLED                   (1)

// Interrupt Mask for PPB_FIFO_OVRF. Generate an interrupt when PPB_FIFO_OVRF
// is set.
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_FIFO_OVRF_INT_MASK                      21:21
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_FIFO_OVRF_INT_MASK_DISABLED                   (0)    // // Don't generate an interrupt when PPB_FIFO_OVRF
// is set.

#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_FIFO_OVRF_INT_MASK_ENABLED                    (1)

// Interrupt Mask for PPB_STMERR. Generate an interrupt when PPB_STMERR
// is set.
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_STMERR_INT_MASK                 22:22
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_STMERR_INT_MASK_DISABLED                      (0)    // // Don't generate an interrupt when PPB_STMERR
// is set.

#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_STMERR_INT_MASK_ENABLED                       (1)

// Interrupt Mask for PPB_SHORT_FRAME. Generate an interrupt when PPB_SHORT_FRAME
// is set.
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_SHORT_FRAME_INT_MASK                    23:23
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_SHORT_FRAME_INT_MASK_DISABLED                 (0)    // // Don't generate an interrupt when PPB_SHORT_FRAME
// is set.

#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_SHORT_FRAME_INT_MASK_ENABLED                  (1)

// Interrupt Mask for PPB_EXTRA_SF. Generate an interrupt when PPB_EXTRA_SF
// is set.
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_EXTRA_SF_INT_MASK                       24:24
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_EXTRA_SF_INT_MASK_DISABLED                    (0)    // // Don't generate an interrupt when PPB_EXTRA_SF
// is set.

#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_EXTRA_SF_INT_MASK_ENABLED                     (1)

// Interrupt Mask for PPB_INTERFRAME_LINE. Generate an interrupt when PPB_INTERFRAME_LINE
// is set.
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_INTERFRAME_LINE_INT_MASK                        25:25
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_INTERFRAME_LINE_INT_MASK_DISABLED                     (0)    // // Don't generate an interrupt when PPB_INTERFRAME_LINE
// is set.

#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_INTERFRAME_LINE_INT_MASK_ENABLED                      (1)

// Interrupt Mask for PPB_SPARE_STATUS_1. Generate an interrupt when PPB_SPARE_STATUS_1
// is set.
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_SPARE_STATUS_1_INT_MASK                 26:26
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_SPARE_STATUS_1_INT_MASK_DISABLED                      (0)    // // Don't generate an interrupt when PPB_SPARE_STATUS_1
// is set.

#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_SPARE_STATUS_1_INT_MASK_ENABLED                       (1)

// Interrupt Mask for PPB_SPARE_STATUS_2. Generate an interrupt when PPB_SPARE_STATUS_2
// is set.
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_SPARE_STATUS_2_INT_MASK                 27:27
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_SPARE_STATUS_2_INT_MASK_DISABLED                      (0)    // // Don't generate an interrupt when PPB_SPARE_STATUS_2
// is set.

#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_PPB_SPARE_STATUS_2_INT_MASK_ENABLED                       (1)

// Interrupt Mask for HPV_UNC_HDR_ERR. Generate an interrupt when HPV_UNC_HDR_ERR
// is set.
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_HPV_UNC_HDR_ERR_INT_MASK                    30:30
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_HPV_UNC_HDR_ERR_INT_MASK_DISABLED                 (0)    // // Don't generate an interrupt when HPV_UNC_HDR_ERR
// is set.

#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_HPV_UNC_HDR_ERR_INT_MASK_ENABLED                  (1)

// Interrupt Mask for HPH_UNC_HDR_ERR. Generate an interrupt when HPH_UNC_HDR_ERR
// is set.
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_HPH_UNC_HDR_ERR_INT_MASK                    31:31
#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_HPH_UNC_HDR_ERR_INT_MASK_DISABLED                 (0)    // // Don't generate an interrupt when HPH_UNC_HDR_ERR
// is set.

#define LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0_HPH_UNC_HDR_ERR_INT_MASK_ENABLED                  (1)


// Register CSI_CSI_CIL_INTERRUPT_MASK_0  // CSI Control and Interface Logic Interrupt Mask
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0                    (0x221)
// Interrupt Mask for CILA_SOT_SB_ERR. Generate an interrupt when CILA_SOT_SB_ERR
// is set.
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILA_SOT_SB_ERR_INT_MASK                     0:0
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILA_SOT_SB_ERR_INT_MASK_DISABLED                  (0)    // // Don't generate an interrupt when CILA_SOT_SB_ERR
// is set.

#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILA_SOT_SB_ERR_INT_MASK_ENABLED                   (1)

// Interrupt Mask for CILA_SOT_MB_ERR. Generate an interrupt when CILA_SOT_MB_ERR
// is set.
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILA_SOT_MB_ERR_INT_MASK                     1:1
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILA_SOT_MB_ERR_INT_MASK_DISABLED                  (0)    // // Don't generate an interrupt when CILA_SOT_MB_ERR
// is set.

#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILA_SOT_MB_ERR_INT_MASK_ENABLED                   (1)

// Interrupt Mask for CILA_SYNC_ESC_ERR. Generate an interrupt when CILA_SYNC_ESC_ERR
// is set.
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILA_SYNC_ESC_ERR_INT_MASK                   2:2
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILA_SYNC_ESC_ERR_INT_MASK_DISABLED                        (0)    // // Don't generate an interrupt when CILA_SYNC_ESC_ERR
// is set.

#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILA_SYNC_ESC_ERR_INT_MASK_ENABLED                 (1)

// Interrupt Mask for CILA_ESC_ENTRY_ERR. Generate an interrupt when CILA_ESC_ENTRY_ERR
// is set.
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILA_ESC_ENTRY_ERR_INT_MASK                  3:3
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILA_ESC_ENTRY_ERR_INT_MASK_DISABLED                       (0)    // // Don't generate an interrupt when CILA_ESC_ENTRY_ERR
// is set.

#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILA_ESC_ENTRY_ERR_INT_MASK_ENABLED                        (1)

// Interrupt Mask for CILA_CTRL_ERR. Generate an interrupt when CILA_CTRL_ERR
// is set.
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILA_CTRL_ERR_INT_MASK                       4:4
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILA_CTRL_ERR_INT_MASK_DISABLED                    (0)    // // Don't generate an interrupt when CILA_CTRL_ERR
// is set.

#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILA_CTRL_ERR_INT_MASK_ENABLED                     (1)

// Interrupt Mask for CILA_ESC_CMD_REC. Generate an interrupt when CILA_ESC_CMD_REC
// is set.
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILA_ESC_CMD_REC_INT_MASK                    5:5
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILA_ESC_CMD_REC_INT_MASK_DISABLED                 (0)    // // Don't generate an interrupt when CILA_ESC_CMD_REC
// is set.

#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILA_ESC_CMD_REC_INT_MASK_ENABLED                  (1)

// Interrupt Mask for CILA_ESC_DATA_REC. Generate an interrupt when CILA_ESC_DATA_REC
// is set.
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILA_ESC_DATA_REC_INT_MASK                   6:6
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILA_ESC_DATA_REC_INT_MASK_DISABLED                        (0)    // // Don't generate an interrupt when CILA_ESC_DATA_REC
// is set.

#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILA_ESC_DATA_REC_INT_MASK_ENABLED                 (1)

// Interrupt Mask for CILA_SPARE_STATUS_1. Generate an interrupt when CILA_SPARE_STATUS_1
// is set.
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILA_SPARE_STATUS_1_INT_MASK                 7:7
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILA_SPARE_STATUS_1_INT_MASK_DISABLED                      (0)    // // Don't generate an interrupt when CILA_SPARE_STATUS_1
// is set.

#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILA_SPARE_STATUS_1_INT_MASK_ENABLED                       (1)

// Interrupt Mask for CILA_SPARE_STATUS_2. Generate an interrupt when CILA_SPARE_STATUS_2
// is set.
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILA_SPARE_STATUS_2_INT_MASK                 8:8
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILA_SPARE_STATUS_2_INT_MASK_DISABLED                      (0)    // // Don't generate an interrupt when CILA_SPARE_STATUS_2
// is set.

#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILA_SPARE_STATUS_2_INT_MASK_ENABLED                       (1)

// Interrupt Mask for MIPI_AUTO_CAL_DONE. Generate an interrupt when MIPI_AUTO_CAL_DONE
// is set.
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_MIPI_AUTO_CAL_DONE_INT_MASK                  15:15
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_MIPI_AUTO_CAL_DONE_INT_MASK_DISABLED                       (0)    // // Don't generate an interrupt when MIPI_AUTO_CAL_DONE
// is set.

#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_MIPI_AUTO_CAL_DONE_INT_MASK_ENABLED                        (1)

// Interrupt Mask for CILB_SOT_SB_ERR. Generate an interrupt when CILB_SOT_SB_ERR
// is set.
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILB_SOT_SB_ERR_INT_MASK                     16:16
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILB_SOT_SB_ERR_INT_MASK_DISABLED                  (0)    // // Don't generate an interrupt when CILB_SOT_SB_ERR
// is set.

#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILB_SOT_SB_ERR_INT_MASK_ENABLED                   (1)

// Interrupt Mask for CILB_SOT_MB_ERR. Generate an interrupt when CILB_SOT_MB_ERR
// is set.
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILB_SOT_MB_ERR_INT_MASK                     17:17
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILB_SOT_MB_ERR_INT_MASK_DISABLED                  (0)    // // Don't generate an interrupt when CILB_SOT_MB_ERR
// is set.

#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILB_SOT_MB_ERR_INT_MASK_ENABLED                   (1)

// Interrupt Mask for CILB_SYNC_ESC_ERR. Generate an interrupt when CILB_SYNC_ESC_ERR
// is set.
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILB_SYNC_ESC_ERR_INT_MASK                   18:18
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILB_SYNC_ESC_ERR_INT_MASK_DISABLED                        (0)    // // Don't generate an interrupt when CILB_SYNC_ESC_ERR
// is set.

#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILB_SYNC_ESC_ERR_INT_MASK_ENABLED                 (1)

// Interrupt Mask for CILB_ESC_ENTRY_ERR. Generate an interrupt when CILB_ESC_ENTRY_ERR
// is set.
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILB_ESC_ENTRY_ERR_INT_MASK                  19:19
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILB_ESC_ENTRY_ERR_INT_MASK_DISABLED                       (0)    // // Don't generate an interrupt when CILB_ESC_ENTRY_ERR
// is set.

#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILB_ESC_ENTRY_ERR_INT_MASK_ENABLED                        (1)

// Interrupt Mask for CILB_CTRL_ERR. Generate an interrupt when CILB_CTRL_ERR
// is set.
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILB_CTRL_ERR_INT_MASK                       20:20
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILB_CTRL_ERR_INT_MASK_DISABLED                    (0)    // // Don't generate an interrupt when CILB_CTRL_ERR
// is set.

#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILB_CTRL_ERR_INT_MASK_ENABLED                     (1)

// Interrupt Mask for CILB_ESC_CMD_REC. Generate an interrupt when CILB_ESC_CMD_REC
// is set.
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILB_ESC_CMD_REC_INT_MASK                    21:21
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILB_ESC_CMD_REC_INT_MASK_DISABLED                 (0)    // // Don't generate an interrupt when CILB_ESC_CMD_REC
// is set.

#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILB_ESC_CMD_REC_INT_MASK_ENABLED                  (1)

// Interrupt Mask for CILB_ESC_DATA_REC. Generate an interrupt when CILB_ESC_DATA_REC
// is set.
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILB_ESC_DATA_REC_INT_MASK                   22:22
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILB_ESC_DATA_REC_INT_MASK_DISABLED                        (0)    // // Don't generate an interrupt when CILB_ESC_DATA_REC
// is set.

#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILB_ESC_DATA_REC_INT_MASK_ENABLED                 (1)

// Interrupt Mask for CILB_SPARE_STATUS_1. Generate an interrupt when CILB_SPARE_STATUS_1
// is set.
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILB_SPARE_STATUS_1_INT_MASK                 23:23
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILB_SPARE_STATUS_1_INT_MASK_DISABLED                      (0)    // // Don't generate an interrupt when CILB_SPARE_STATUS_1
// is set.

#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILB_SPARE_STATUS_1_INT_MASK_ENABLED                       (1)

// Interrupt Mask for CILB_SPARE_STATUS_2. Generate an interrupt when CILB_SPARE_STATUS_2
// is set.
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILB_SPARE_STATUS_2_INT_MASK                 24:24
#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILB_SPARE_STATUS_2_INT_MASK_DISABLED                      (0)    // // Don't generate an interrupt when CILB_SPARE_STATUS_2
// is set.

#define LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0_CILB_SPARE_STATUS_2_INT_MASK_ENABLED                       (1)


// Register CSI_CSI_READONLY_STATUS_0  // CSI Read Only Status, this register is used to return
// CSI read only status.
#define LWE24D_CSI_CSI_READONLY_STATUS_0                       (0x222)
// One only when Pixel Parser A is capturing frame data.
#define LWE24D_CSI_CSI_READONLY_STATUS_0_CSI_PPA_ACTIVE                  0:0

// One only when Pixel Parser B is capturing frame data.
#define LWE24D_CSI_CSI_READONLY_STATUS_0_CSI_PPB_ACTIVE                  1:1

// Reads back CSI's interrupt line. This is being used test
// the CSI logic that generates interrupt.
#define LWE24D_CSI_CSI_READONLY_STATUS_0_CSI_RO_SPARE2                   2:2

// Spare Read Only status bit
#define LWE24D_CSI_CSI_READONLY_STATUS_0_CSI_RO_SPARE3                   3:3

// Spare Read Only status bit
#define LWE24D_CSI_CSI_READONLY_STATUS_0_CSI_RO_SPARE4                   4:4

// Spare Read Only status bit
#define LWE24D_CSI_CSI_READONLY_STATUS_0_CSI_RO_SPARE5                   5:5

// Spare Read Only status bit
#define LWE24D_CSI_CSI_READONLY_STATUS_0_CSI_RO_SPARE6                   6:6

// Spare Read Only status bit
#define LWE24D_CSI_CSI_READONLY_STATUS_0_CSI_RO_SPARE7                   7:7


// Register CSI_ESCAPE_MODE_COMMAND_0  // Escape Mode Command, this register is used to receive
// escape mode command bytes from CIL-A and CIL-B.
#define LWE24D_CSI_ESCAPE_MODE_COMMAND_0                       (0x223)
// CIL-A Escape Mode Command Byte, this is the 8 bit entry
// command that was received, by CIL-A, during the last 
// escape Mode sequence. CIL-A monitors Byte Lane 0, only,
// for escape mode sequences. This command byte can only 
// be  assummed to be valid when CILA_ESC_CMD_REC status
// bit is set.
#define LWE24D_CSI_ESCAPE_MODE_COMMAND_0_CILA_ESC_CMD_BYTE                       7:0

// CIL-B Escape Mode Command Byte, this is the 8 bit entry
// command that was received, by CIL-B, during the last 
// escape Mode sequence. This command byte can only be 
// assummed to be valid when CILB_ESC_CMD_REC status bit
// is set.
#define LWE24D_CSI_ESCAPE_MODE_COMMAND_0_CILB_ESC_CMD_BYTE                       23:16


// Register CSI_ESCAPE_MODE_DATA_0  // Escape Mode Data, this register is used to receive
// escape mode data bytes from CIL-A and CIL-B.
#define LWE24D_CSI_ESCAPE_MODE_DATA_0                  (0x224)
// CIL-A Escape Mode Data Byte, when read this field returns
// the last Escape Mode Data byte that was received by CIL-A.
// Escape Mode Data bytes are the bytes that are received
// in Escape Mode after receiving the Escape Mode Command.
// These bytes can be used to implement MIPI's CSI Specs Low
// Power Data Transmition. This field is only valid when 
// the status bit, CILA_ESC_DATA_REC, is set, and will be
// overwritten by the next Escape Mode data byte if not read
// before the next byte come in.
#define LWE24D_CSI_ESCAPE_MODE_DATA_0_CILA_ESC_DATA_BYTE                 7:0

// CIL-B Escape Mode Data Byte, when read this field returns
// the last Escape Mode Data byte that was received by CIL-B.
// Escape Mode Data bytes are the bytes that are received
// in Escape Mode after receiving the Escape Mode Command.
// These bytes can be used to implement MIPI's CSI Specs Low
// Power Data Transmition. This field is only valid when 
// the status bit, CILB_ESC_DATA_REC, is set, and will be
// overwritten by the next Escape Mode data byte if not read
// before the next byte come in.
#define LWE24D_CSI_ESCAPE_MODE_DATA_0_CILB_ESC_DATA_BYTE                 23:16


// Register CSI_CILA_PAD_CONFIG0_0  // CIL-A Pad Configuration 0
#define LWE24D_CSI_CILA_PAD_CONFIG0_0                  (0x225)
// Power down for each data bit, including drivers,
// receivers and contention detectors
#define LWE24D_CSI_CILA_PAD_CONFIG0_0_PAD_CILA_PDIO                      1:0

// Power down for clock bit, including drivers, 
// receivers and contention detectors
#define LWE24D_CSI_CILA_PAD_CONFIG0_0_PAD_CILA_PDIO_CLK                  2:2

// HS driver preemphasis enable,1= preemphasis enabled
#define LWE24D_CSI_CILA_PAD_CONFIG0_0_PAD_CILA_PREEMP_EN                 3:3

// Clock bit input delay trimmer, each tap delays 20ps
#define LWE24D_CSI_CILA_PAD_CONFIG0_0_PAD_CILA_INADJCLK                  6:4

// bit 0 input delay trimmer, each tap delays 20ps
#define LWE24D_CSI_CILA_PAD_CONFIG0_0_PAD_CILA_INADJ0                    10:8

// bit 1 input delay trimmer, each tap delays 20ps
#define LWE24D_CSI_CILA_PAD_CONFIG0_0_PAD_CILA_INADJ1                    14:12

// Increase bandwidth of differential receiver
#define LWE24D_CSI_CILA_PAD_CONFIG0_0_PAD_CILA_BANDWD_IN                 16:16

// Driver pull up impedance control
// 00 -> 130ohm, default
// 01 -> 110ohm
// 10 -> 130ohm, same as 00
// 11 -> 150ohm
#define LWE24D_CSI_CILA_PAD_CONFIG0_0_PAD_CILA_LPUPADJ                   21:20

// Driver pull down impedance control
// 00 -> 130ohm, default
// 01 -> 110ohm
// 10 -> 130ohm, same as 00
// 11 -> 150ohm
#define LWE24D_CSI_CILA_PAD_CONFIG0_0_PAD_CILA_LPDNADJ                   23:22

// Pull up slew rate adjust, default 000
// From 000 -> 011, slew rate increases
// 100 is the same as 000
// From 100->111, skew rate decreases.
#define LWE24D_CSI_CILA_PAD_CONFIG0_0_PAD_CILA_SLEWUPADJ                 26:24

// Pull down slew rate adjust, default 000
// From 000 -> 011, slew rate increases
// 100 is the same as 000
// From 100->111, skew rate decreases.
#define LWE24D_CSI_CILA_PAD_CONFIG0_0_PAD_CILA_SLEWDNADJ                 30:28


// Register CSI_CILA_PAD_CONFIG1_0  // CIL-A Pad Configuration 4
#define LWE24D_CSI_CILA_PAD_CONFIG1_0                  (0x226)
// Spare bits for CILA Config
// PAD_CILA_SPARE[15] is being used to disable 
// the CSI-A RTL code that blocks fifo pushs 
// that are past the end of the line packet.
// 0: disabled, 1: push blocking enabled
#define LWE24D_CSI_CILA_PAD_CONFIG1_0_PAD_CILA_SPARE                     15:0

// Spare Read only bits for CILA Config
#define LWE24D_CSI_CILA_PAD_CONFIG1_0_PAD_CILA_SPARE_RO                  31:16


// Register CSI_CILB_PAD_CONFIG0_0  // CIL-B Pad Configuration 0
#define LWE24D_CSI_CILB_PAD_CONFIG0_0                  (0x227)
// Power down for each data bit, including drivers,
// receivers and contention detectors
#define LWE24D_CSI_CILB_PAD_CONFIG0_0_PAD_CILB_PDIO                      0:0

// Power down for clock bit, including drivers, 
// receivers and contention detectors
#define LWE24D_CSI_CILB_PAD_CONFIG0_0_PAD_CILB_PDIO_CLK                  2:2

// HS driver preemphasis enable,1= preemphasis enabled
#define LWE24D_CSI_CILB_PAD_CONFIG0_0_PAD_CILB_PREEMP_EN                 3:3

// Clock bit input delay trimmer, each tap delays 20ps
#define LWE24D_CSI_CILB_PAD_CONFIG0_0_PAD_CILB_INADJCLK                  6:4

// bit 0 input delay trimmer, each tap delays 20ps
#define LWE24D_CSI_CILB_PAD_CONFIG0_0_PAD_CILB_INADJ0                    10:8

// Increase bandwidth of differential receiver
#define LWE24D_CSI_CILB_PAD_CONFIG0_0_PAD_CILB_BANDWD_IN                 16:16

// Driver pull up impedance control
// 00 -> 130ohm, default
// 01 -> 110ohm
// 10 -> 130ohm, same as 00
// 11 -> 150ohm
#define LWE24D_CSI_CILB_PAD_CONFIG0_0_PAD_CILB_LPUPADJ                   21:20

// Driver pull down impedance control
// 00 -> 130ohm, default
// 01 -> 110ohm
// 10 -> 130ohm, same as 00
// 11 -> 150ohm
#define LWE24D_CSI_CILB_PAD_CONFIG0_0_PAD_CILB_LPDNADJ                   23:22

// Pull up slew rate adjust, default 000
// From 000 -> 011, slew rate increases
// 100 is the same as 000
// From 100->111, skew rate decreases.
#define LWE24D_CSI_CILB_PAD_CONFIG0_0_PAD_CILB_SLEWUPADJ                 26:24

// Pull down slew rate adjust, default 000
// From 000 -> 011, slew rate increases
// 100 is the same as 000
// From 100->111, skew rate decreases.
#define LWE24D_CSI_CILB_PAD_CONFIG0_0_PAD_CILB_SLEWDNADJ                 30:28


// Register CSI_CILB_PAD_CONFIG1_0  // CIL-B Pad Configuration 4
#define LWE24D_CSI_CILB_PAD_CONFIG1_0                  (0x228)
// Spare bits for CILB Config
// PAD_CILB_SPARE[15] is being used to disable 
// the CSI-B RTL code that blocks fifo pushs 
// that are past the end of the line packet.
// 0: disabled, 1: push blocking enabled
#define LWE24D_CSI_CILB_PAD_CONFIG1_0_PAD_CILB_SPARE                     15:0

// Spare Read only bits for CILB Config
#define LWE24D_CSI_CILB_PAD_CONFIG1_0_PAD_CILB_SPARE_RO                  31:16


// Register CSI_CIL_PAD_CONFIG0_0  // CIL Pad Configuration 0
#define LWE24D_CSI_CIL_PAD_CONFIG0_0                   (0x229)
// Bypass bang gap voltage reference
#define LWE24D_CSI_CIL_PAD_CONFIG0_0_PAD_CIL_VBYPASS                     0:0

// Power down voltage regulator, 1=power down
#define LWE24D_CSI_CIL_PAD_CONFIG0_0_PAD_CIL_PDVREG                      1:1

// VAUXP level adjustment
// 00 -> no adjustment, default
// 01 -> 105% 
// 10 -> 110% 
// 11 -> 115%
// 100 -> no adjustment
// 101 -> 95%
// 110 -> 90%
// 111 -> 85%
#define LWE24D_CSI_CIL_PAD_CONFIG0_0_PAD_CIL_VADJ                        6:4

// Spare bit for CIL BIAS Config
// PAD_CIL_SPARE[7] is used is being used to flush VI's
// Y-FIFO when it is being use as a stream source for 
// one of the Pixel Parsers. Setting PAD_CIL_SPARE[7]
// to 1 will hold vi2csi_host_stall low. Which will
// force VI's Y-FIFO to be purged. PAD_CIL_SPARE[7]
// must be low for the pixel parser to receive source
// data from VI's Y-FIFO. 
#define LWE24D_CSI_CIL_PAD_CONFIG0_0_PAD_CIL_SPARE                       15:8


// Register CSI_CILA_MIPI_CAL_CONFIG_0  // Calibration settings for CIL-A mipi pads
#define LWE24D_CSI_CILA_MIPI_CAL_CONFIG_0                      (0x22a)
// 2's complement offset for TERMADJ going to channel A
#define LWE24D_CSI_CILA_MIPI_CAL_CONFIG_0_MIPI_CAL_TERMOSA                       4:0

// 2's complement offset for HSPUADJ going to channel A
#define LWE24D_CSI_CILA_MIPI_CAL_CONFIG_0_MIPI_CAL_HSPUOSA                       12:8

// 2's complement offset for HSPDADJ going to channel A
#define LWE24D_CSI_CILA_MIPI_CAL_CONFIG_0_MIPI_CAL_HSPDOSA                       20:16

// Select the CSIA PADS for auto calibration.
#define LWE24D_CSI_CILA_MIPI_CAL_CONFIG_0_MIPI_CAL_SELA                  21:21

// Auto Cal calibration step prescale:
// Set to 00 when calibration step should be 0.1 us
// Set to 01 when calibration step should be 0.5 us
// Set to 10 when calibration step should be 1.0 us
// Set to 11 when calibration step should be 1.5 us
// this will keep the mipi bias cal step between 0.1-1.5 usec
// Default set for 1.0 us calibraiton step.
#define LWE24D_CSI_CILA_MIPI_CAL_CONFIG_0_MIPI_CAL_PRESCALE                      25:24

// The DRIVRY & TERMRY signals coming from MIPI Pads are
// utilized by Calibration state machine for PAD Calibration.
// The drivry/termry comes from a noisy analog source 
// and it could have some glitches.
// The filter in calibsm is sensitive to these noises.
// If the calibration done status does not show up, we
// can change the sensitivity of the filter through these bits.
// Ideally this has to be programmed in a range from 10 to 15.
// For the case when MIPI_CAL_PRESCALE = 2'b00, this needs to be
// programmed between 2 to 5.
#define LWE24D_CSI_CILA_MIPI_CAL_CONFIG_0_MIPI_CAL_NOISE_FLT                     29:26

// When 0 (normal operation), use the above registers
// as an offset to the Calibration State machine setting
// for channel A TERMADJ/HSPUADJ/HSPDADJ values to the 
// Mipi Pads. When 1, use the register values above as
// the actual value going to channel A TERMADJ/HSPUADJ/HSPDADJ
// on the Mipi Pads.
#define LWE24D_CSI_CILA_MIPI_CAL_CONFIG_0_MIPI_CAL_OVERIDEA                      30:30

// Writting a one to this bit starts the Calibration State
// machine.  This bit must be set even if both overrides
// set in order to latch in the over ride value
#define LWE24D_CSI_CILA_MIPI_CAL_CONFIG_0_MIPI_CAL_STARTCAL                      31:31


// Register CSI_CILB_MIPI_CAL_CONFIG_0  // Calibration settings for CIL-B mipi pads
#define LWE24D_CSI_CILB_MIPI_CAL_CONFIG_0                      (0x22b)
// 2's complement offset for TERMADJ going to channel B
#define LWE24D_CSI_CILB_MIPI_CAL_CONFIG_0_MIPI_CAL_TERMOSB                       4:0

// 2's complement offset for HSPUADJ going to channel B
#define LWE24D_CSI_CILB_MIPI_CAL_CONFIG_0_MIPI_CAL_HSPUOSB                       12:8

// 2's complement offset for HSPDADJ going to channel B
#define LWE24D_CSI_CILB_MIPI_CAL_CONFIG_0_MIPI_CAL_HSPDOSB                       20:16

// Select the CSIB PADS for auto calibration.
#define LWE24D_CSI_CILB_MIPI_CAL_CONFIG_0_MIPI_CAL_SELB                  21:21

// When 0 (normal operation), use the above registers
// as an offset to the Calibration State machine setting
// for channel B TERMADJ/HSPUADJ/HSPDADJ values to the 
// Mipi Pads. When 1, use the register values above as
// the actual value going to channel B TERMADJ/HSPUADJ/HSPDADJ
// on the Mipi Pads.
// Writting a one to Bit 31 of CILA_MIPI_CAL_CONFIG 
// (MIPI_CAL_STARTCAL) starts the Calibration State
// machine.
#define LWE24D_CSI_CILB_MIPI_CAL_CONFIG_0_MIPI_CAL_OVERIDEB                      30:30


// Register CSI_CIL_MIPI_CAL_STATUS_0  // CIL MIPI Calibrate Status
#define LWE24D_CSI_CIL_MIPI_CAL_STATUS_0                       (0x22c)
// One when auto calibrate is active.
#define LWE24D_CSI_CIL_MIPI_CAL_STATUS_0_MIPI_CAL_ACTIVE                 0:0

// Termination code generated by MIPI auto Calibrate.
// Valid only after auto calibrate sequence has 
// completed (MIPI_CAL_ACTIVE == 0).
#define LWE24D_CSI_CIL_MIPI_CAL_STATUS_0_MIPI_CAL_TERMADJ                        7:4

// Driver code generated by MIPI auto Calibrate.
// Valid only after auto calibrate sequence has 
// completed (MIPI_CAL_ACTIVE == 0).
#define LWE24D_CSI_CIL_MIPI_CAL_STATUS_0_MIPI_CAL_DRIVADJ                        11:8

// MIPI Auto Calibrate done for CSI,
// set when the auto calibrate 
// sequence for CSI pad bricks is done.
#define LWE24D_CSI_CIL_MIPI_CAL_STATUS_0_MIPI_AUTO_CAL_DONE_CSIA                 27:27

// MIPI Auto Calibrate done for CSI,
// set when the auto calibrate 
// sequence for CSI pad bricks is done.
#define LWE24D_CSI_CIL_MIPI_CAL_STATUS_0_MIPI_AUTO_CAL_DONE_CSIB                 28:28

// MIPI Auto Calibrate done for DSI,
// set when the auto calibrate 
// sequence for DSI pad bricks is done.
#define LWE24D_CSI_CIL_MIPI_CAL_STATUS_0_MIPI_AUTO_CAL_DONE_DSI                  29:29

// Second-level clock enable override register
//
// This can override the 2nd level clock enables in case of malfunction.
// Only exposed to software when needed.
//

// Register CSI_CLKEN_OVERRIDE_0  
#define LWE24D_CSI_CLKEN_OVERRIDE_0                    (0x22d)
#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_CLKEN_OVR                        0:0
#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_CLKEN_OVR_CLK_GATED                    (0)
#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_CLKEN_OVR_CLK_ALWAYS_ON                        (1)

#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_DBG_CLKEN_OVR                    1:1
#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_DBG_CLKEN_OVR_CLK_GATED                        (0)
#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_DBG_CLKEN_OVR_CLK_ALWAYS_ON                    (1)

#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_FV_CLKEN_OVR                     2:2
#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_FV_CLKEN_OVR_CLK_GATED                 (0)
#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_FV_CLKEN_OVR_CLK_ALWAYS_ON                     (1)

#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_FA_CLKEN_OVR                     3:3
#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_FA_CLKEN_OVR_CLK_GATED                 (0)
#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_FA_CLKEN_OVR_CLK_ALWAYS_ON                     (1)

#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_FB_CLKEN_OVR                     4:4
#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_FB_CLKEN_OVR_CLK_GATED                 (0)
#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_FB_CLKEN_OVR_CLK_ALWAYS_ON                     (1)

#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_HPV_CLKEN_OVR                    5:5
#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_HPV_CLKEN_OVR_CLK_GATED                        (0)
#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_HPV_CLKEN_OVR_CLK_ALWAYS_ON                    (1)

#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_HPA_CLKEN_OVR                    6:6
#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_HPA_CLKEN_OVR_CLK_GATED                        (0)
#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_HPA_CLKEN_OVR_CLK_ALWAYS_ON                    (1)

#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_HPB_CLKEN_OVR                    7:7
#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_HPB_CLKEN_OVR_CLK_GATED                        (0)
#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_HPB_CLKEN_OVR_CLK_ALWAYS_ON                    (1)

#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_HPH_CLKEN_OVR                    8:8
#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_HPH_CLKEN_OVR_CLK_GATED                        (0)
#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_HPH_CLKEN_OVR_CLK_ALWAYS_ON                    (1)

#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_PPA_CLKEN_OVR                    9:9
#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_PPA_CLKEN_OVR_CLK_GATED                        (0)
#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_PPA_CLKEN_OVR_CLK_ALWAYS_ON                    (1)

#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_PPB_CLKEN_OVR                    10:10
#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_PPB_CLKEN_OVR_CLK_GATED                        (0)
#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_PPB_CLKEN_OVR_CLK_ALWAYS_ON                    (1)

#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_CILA_CLKEN_OVR                   11:11
#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_CILA_CLKEN_OVR_CLK_GATED                       (0)
#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_CILA_CLKEN_OVR_CLK_ALWAYS_ON                   (1)

#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_CILB_CLKEN_OVR                   12:12
#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_CILB_CLKEN_OVR_CLK_GATED                       (0)
#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_CILB_CLKEN_OVR_CLK_ALWAYS_ON                   (1)

#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_CIL_CLKEN_OVR                    13:13
#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_CIL_CLKEN_OVR_CLK_GATED                        (0)
#define LWE24D_CSI_CLKEN_OVERRIDE_0_CSI_CIL_CLKEN_OVR_CLK_ALWAYS_ON                    (1)


// Register CSI_DEBUG_CONTROL_0  // Debug Control
#define LWE24D_CSI_DEBUG_CONTROL_0                     (0x22e)
// Debug Enable Second level CSI Debug clock is enabled. Debug counters
// 2, 1 & 0 are powered up.
#define LWE24D_CSI_DEBUG_CONTROL_0_DEBUG_EN                      0:0
#define LWE24D_CSI_DEBUG_CONTROL_0_DEBUG_EN_DISABLED                   (0)    // // Debug counters 2, 1 & 0 are powered down. Second level
// CSI Debug clock is disabled.

#define LWE24D_CSI_DEBUG_CONTROL_0_DEBUG_EN_ENABLED                    (1)

// When CSI-A is operating in a "Header Not Sent mode",
// writting a 1 to this bit indicates start frame (SF)
// or end frame (EF) control code. After the pixel parser 
// is enabled, writing a 1 to this bit will start frame 
// capture and send start frame (SF) control code. Writing
// a 1 to this bit again will stop frame capture and send
// end frame (EF) control code. "Header Not Sent mode" can 
// be used as a debug mode to capture what the sensor
// is sending without interpeting the packets. Writing a
// 1 to this bit continually will generate SF and EF control
// codes. Note that a wait for MISC_CSI_PPA_FRAME_END syncpt
// is needed between an EF trigger for the current frame and 
// an SF trigger for the next frame.
#define LWE24D_CSI_DEBUG_CONTROL_0_CSIA_DBG_SF                   2:2

// When CSI-B is operating in a "Header Not Sent mode",
// writting a 1 to this bit indicates start frame (SF)
// or end frame (EF) control code. After the pixel parser 
// is enabled, writing a 1 to this bit will start frame 
// capture and send start frame (SF) control code. Writing
// a 1 to this bit again will stop frame capture and send
// end frame (EF) control code. "Header Not Sent mode" can 
// be used as a debug mode to capture what the sensor
// is sending without interpeting the packets. Writing a
// 1 to this bit continually will generate SF and EF control
// codes. Note that a wait for MISC_CSI_PPB_FRAME_END syncpt
// is needed between an EF trigger for the current frame and 
// an SF trigger for the next frame.
#define LWE24D_CSI_DEBUG_CONTROL_0_CSIB_DBG_SF                   3:3

// Clear Debug Counter 0, write a one to this bit to clear
// debug counter 0 and dbg_cnt_rolled_0.
#define LWE24D_CSI_DEBUG_CONTROL_0_CLR_DBG_CNT_0                 4:4

// Clear Debug Counter 1, write a one to this bit to clear
// debug counter 1 and dbg_cnt_rolled_1.
#define LWE24D_CSI_DEBUG_CONTROL_0_CLR_DBG_CNT_1                 5:5

// Clear Debug Counter 2, write a one to this bit to clear
// debug counter 2 and dbg_cnt_rolled_2.
#define LWE24D_CSI_DEBUG_CONTROL_0_CLR_DBG_CNT_2                 6:6

// Debug Count Select 0, this field selects what will be 
// counted by debug counter 0.
// Encodings 00 to 31 selects the set signal for one of 
// the CSI_PIXEL_PARSER_STATUS register bits. In this case
// the select encoding of this field is the same as the
// bit position, in CSI_PIXEL_PARSER_STATUS, of the status
// bit who's set signal will be used to increment DBG_CNT_0.
// Encodings 32 to 63 selects the set signal for one of 
// the CSI_CIL_STATUS status bits. The least significant 
// 5 bits of this select field give the bit position, in
// CSI_CIL_STATUS, of the status bit whos set signal pluses
// will be counted by dbg_cnt_0. Selections for encodings 64
// to 127 are given below: 
// 64 - PPA Line packets processed
// 65 - PPA short packets processed
// 66 - Total packets processed by PPA
// 67 - PPA Frame Starts Outputted
// 68 - PPA Frame Ends Outputted
// 69 - Reserved encoding
// 70 - PPB Line packets processed
// 71 - PPB short packets processed
// 72 - Total packets processed by PPB
// 73 - PPB Frame Starts Outputted
// 74 - PPB Frame Ends Outputted
// 75 - Reserved encoding
// 76 - HPA Headers Parsed
// 77 - HPA Headers Parsed with no ECC Errors
// 78 - HPB Headers Parsed
// 79 - HPB Headers Parsed with no ECC Errors
// 80 - HPV Headers Parsed
// 81 - HPV Headers Parsed with no ECC Errors
// 82 - HPH Headers Parsed
// 83 - HPH Headers Parsed with no ECC Errors
// 84 - 32 bit words read from vi2csi_host_data
// 85 to 127 - Reserved encodings
#define LWE24D_CSI_DEBUG_CONTROL_0_DBG_CNT_SEL_0                 14:8

// Set when dbg_cnt_0 is incremented past max count, cleared
// when clr_dbg_cnt_0 is written with a value of 1.
#define LWE24D_CSI_DEBUG_CONTROL_0_DBG_CNT_ROLLED_0                      15:15

// Debug Count Select 1, this field selects what will be 
// counted by debug counter 1.
// Encodings 00 to 31 selects the set signal for one of 
// the CSI_PIXEL_PARSER_STATUS register bits. In this case
// the select encoding of this field is the same as the
// bit position, in CSI_PIXEL_PARSER_STATUS, of the status
// bit who's set signal will be used to increment DBG_CNT_0.
// Encodings 32 to 63 selects the set signal for one of 
// the CSI_CIL_STATUS status bits. The least significant 
// 5 bits of this select field give the bit position, in
// CSI_CIL_STATUS, of the status bit whos set signal pluses
// will be counted by dbg_cnt_0. Selections for encodings 64
// to 127 are given below: 
// 64 - PPA Line packets processed
// 65 - PPA short packets processed
// 66 - Total packets processed by PPA
// 67 - PPA Frame Starts Outputted
// 68 - PPA Frame Ends Outputted
// 69 - Reserved encoding
// 70 - PPB Line packets processed
// 71 - PPB short packets processed
// 72 - Total packets processed by PPB
// 73 - PPB Frame Starts Outputted
// 74 - PPB Frame Ends Outputted
// 75 - Reserved encoding
// 76 - HPA Headers Parsed
// 77 - HPA Headers Parsed with no ECC Errors
// 78 - HPB Headers Parsed
// 79 - HPB Headers Parsed with no ECC Errors
// 80 - HPV Headers Parsed
// 81 - HPV Headers Parsed with no ECC Errors
// 82 - HPH Headers Parsed
// 83 - HPH Headers Parsed with no ECC Errors
// 84 - 32 bit words read from vi2csi_host_data
// 85 to 127 - Reserved encodings
#define LWE24D_CSI_DEBUG_CONTROL_0_DBG_CNT_SEL_1                 22:16

// Set when dbg_cnt_1 is incremented past max count, cleared
// when clr_dbg_cnt_1 is written with a value of 1.
#define LWE24D_CSI_DEBUG_CONTROL_0_DBG_CNT_ROLLED_1                      23:23

// Debug Count Select 2, this field selects what will be 
// counted by debug counter 2.
// Encodings 00 to 31 selects the set signal for one of 
// the CSI_PIXEL_PARSER_STATUS register bits. In this case
// the select encoding of this field is the same as the
// bit position, in CSI_PIXEL_PARSER_STATUS, of the status
// bit who's set signal will be used to increment DBG_CNT_0.
// Encodings 32 to 63 selects the set signal for one of 
// the CSI_CIL_STATUS status bits. The least significant 
// 5 bits of this select field give the bit position, in
// CSI_CIL_STATUS, of the status bit whos set signal pluses
// will be counted by dbg_cnt_0. Selections for encodings 64
// to 127 are given below: 
// 64 - PPA Line packets processed
// 65 - PPA short packets processed
// 66 - Total packets processed by PPA
// 67 - PPA Frame Starts Outputted
// 68 - PPA Frame Ends Outputted
// 69 - Reserved encoding
// 70 - PPB Line packets processed
// 71 - PPB short packets processed
// 72 - Total packets processed by PPB
// 73 - PPB Frame Starts Outputted
// 74 - PPB Frame Ends Outputted
// 75 - Reserved encoding
// 76 - HPA Headers Parsed
// 77 - HPA Headers Parsed with no ECC Errors
// 78 - HPB Headers Parsed
// 79 - HPB Headers Parsed with no ECC Errors
// 80 - HPV Headers Parsed
// 81 - HPV Headers Parsed with no ECC Errors
// 82 - HPH Headers Parsed
// 83 - HPH Headers Parsed with no ECC Errors
// 84 - 32 bit words read from vi2csi_host_data
// 85 to 127 - Reserved encodings
#define LWE24D_CSI_DEBUG_CONTROL_0_DBG_CNT_SEL_2                 30:24

// Set when dbg_cnt_2 is incremented past max count, cleared
// when clr_dbg_cnt_2 is written with a value of 1.
#define LWE24D_CSI_DEBUG_CONTROL_0_DBG_CNT_ROLLED_2                      31:31

// Register CSI_DEBUG_COUNTER_0_0  // Debug Counter 0, this register can be used to count 
// error conditions or packets processed.
#define LWE24D_CSI_DEBUG_COUNTER_0_0                   (0x22f)
// When read returns the value of debug counter 0.
#define LWE24D_CSI_DEBUG_COUNTER_0_0_DBG_CNT_0                   31:0

// Register CSI_DEBUG_COUNTER_1_0  // Debug Counter 1, this register can be used to count 
// error conditions or packets processed.
#define LWE24D_CSI_DEBUG_COUNTER_1_0                   (0x230)
// When read returns the value of debug counter 1.
#define LWE24D_CSI_DEBUG_COUNTER_1_0_DBG_CNT_1                   31:0

// Register CSI_DEBUG_COUNTER_2_0  // Debug Counter 2, this register can be used to count 
// error conditions or packets processed.
#define LWE24D_CSI_DEBUG_COUNTER_2_0                   (0x231)
// When read returns the value of debug counter 2.
#define LWE24D_CSI_DEBUG_COUNTER_2_0_DBG_CNT_2                   31:0

// Register CSI_PIXEL_STREAM_A_EXPECTED_FRAME_0  // CSI Pixel Stream A Expected Frame
#define LWE24D_CSI_PIXEL_STREAM_A_EXPECTED_FRAME_0                     (0x232)
// When set to one enables checking of the time between
// start line requests from the Header Parser to CSI-PPA.
// A fake EF will be outputted by CSI-PPA if this time 
// between line starts exceeds the value in 
// MAX_CLOCKS_BETWEEN_LINES. Padding lines can be inserted
// before the fake EF, if the number of lines outputted,
// when the fake EF is generated is less than the expected
// frame height. The type of padding is specified using
// CSI_PPA_PAD_FRAME.
#define LWE24D_CSI_PIXEL_STREAM_A_EXPECTED_FRAME_0_PPA_ENABLE_LINE_TIMEOUT                       0:0

// Maximum Number of viclk clock cycles between line
// start requests. The value in this field is in terms
// of 256 viclk clock cycles.
#define LWE24D_CSI_PIXEL_STREAM_A_EXPECTED_FRAME_0_PPA_MAX_CLOCKS                        15:4

// CSI-PPA Expected Frame Height
// Specifies the expected height of the CSI-PPA frame 
// output. Padding out of frames that are shorter
// than this expected height can be specified using
// CSI_PPA_PAD_FRAME. If CSI_PPA_PAD_FRAME is set to
// PAD0S or PAD1S, this parameter must be programmed.
// If CSI_PPA_PAD_FRAME is set to NOPAD, this parameter
// may not be programmed. 
// Programmed Value = number of lines
#define LWE24D_CSI_PIXEL_STREAM_A_EXPECTED_FRAME_0_PPA_EXP_FRAME_HEIGHT                  28:16

// Register CSI_PIXEL_STREAM_B_EXPECTED_FRAME_0  // CSI Pixel Stream B Expected Frame
#define LWE24D_CSI_PIXEL_STREAM_B_EXPECTED_FRAME_0                     (0x233)
// When set to one enables checking of the time between
// start line requests from the Header Parser to CSI-PPB.
// A fake EF will be outputted by CSI-PPB if this time 
// between line starts exceeds the value in 
// MAX_CLOCKS_BETWEEN_LINES. Padding lines can be inserted
// before the fake EF, if the number of lines outputted,
// when the fake EF is generated is less than the expected
// frame height. The type of padding is specified using
// CSI_PPB_PAD_FRAME.
#define LWE24D_CSI_PIXEL_STREAM_B_EXPECTED_FRAME_0_PPB_ENABLE_LINE_TIMEOUT                       0:0

// Maximum Number of viclk clock cycles between line
// start requests. The value in this field is in terms
// of 256 viclk clock cycles.
#define LWE24D_CSI_PIXEL_STREAM_B_EXPECTED_FRAME_0_PPB_MAX_CLOCKS                        15:4

// CSI-PPB Expected Frame Height
// Specifies the expected height of the CSI-PPB frame 
// output. Padding out of frames that are shorter
// than this expected height can be specified using
// CSI_PPB_PAD_FRAME. If CSI_PPB_PAD_FRAME is set to
// PAD0S or PAD1S, this parameter must be programmed.
// If CSI_PPB_PAD_FRAME is set to NOPAD, this parameter
// may not be programmed.
// Programmed Value = number of lines
#define LWE24D_CSI_PIXEL_STREAM_B_EXPECTED_FRAME_0_PPB_EXP_FRAME_HEIGHT                  28:16

// Register CSI_DSI_MIPI_CAL_CONFIG_0  // Calibration settings for DSI mipi pad
#define LWE24D_CSI_DSI_MIPI_CAL_CONFIG_0                       (0x234)
// 2's complement offset for TERMADJ
#define LWE24D_CSI_DSI_MIPI_CAL_CONFIG_0_MIPI_CAL_TERMOSD                        4:0

// 2's complement offset for HSPUADJ
#define LWE24D_CSI_DSI_MIPI_CAL_CONFIG_0_MIPI_CAL_HSPUOSD                        12:8

// 2's complement offset for HSPDADJ
#define LWE24D_CSI_DSI_MIPI_CAL_CONFIG_0_MIPI_CAL_HSPDOSD                        20:16

// Select the DSI PADS for auto calibration.
#define LWE24D_CSI_DSI_MIPI_CAL_CONFIG_0_MIPI_CAL_SELD                   21:21

// When 0 (normal operation), use the above registers
// as an offset to the Calibration State machine setting
// for TERMADJ/HSPUADJ/HSPDADJ values to the 
// Mipi Pads. When 1, use the register values above as
// the actual value going to TERMADJ/HSPUADJ/HSPDADJ
// on the Mipi Pads.
// Writting a one to Bit 31 of CILA_MIPI_CAL_CONFIG 
// (MIPI_CAL_STARTCAL) starts the Calibration State
// machine.
#define LWE24D_CSI_DSI_MIPI_CAL_CONFIG_0_MIPI_CAL_OVERIDED                       30:30

//Interface packets
//SENSOR2CIL

// Packet SENSOR2CIL_PKT
#define SENSOR2CIL_PKT_SIZE 10

// Data
#define SENSOR2CIL_PKT_BYTE                       (7):(0)
#define SENSOR2CIL_PKT_BYTE_ROW                 0

// Start of frame
#define SENSOR2CIL_PKT_SOT                        (8):(8)
#define SENSOR2CIL_PKT_SOT_ROW                  0

// End of frame
#define SENSOR2CIL_PKT_EOT                        (9):(9)
#define SENSOR2CIL_PKT_EOT_ROW                  0

//CIL2CSI

// Packet CIL2CSI_PKT
#define CIL2CSI_PKT_SIZE 8

// Data
#define CIL2CSI_PKT_BYTE                  (7):(0)
#define CIL2CSI_PKT_BYTE_ROW                    0

//VI2CSI_HOST

// Packet VI2CSI_HOST_PKT
#define VI2CSI_HOST_PKT_SIZE 33

// Data
#define VI2CSI_HOST_PKT_HOSTDATA                  (31):(0)
#define VI2CSI_HOST_PKT_HOSTDATA_ROW                    0

// End of packet tag, 0: end of packet, 1: valid packet data
#define VI2CSI_HOST_PKT_TAG                       (32):(32)
#define VI2CSI_HOST_PKT_TAG_ROW                 0

// VI2CSI_VIP

// Packet VI2CSI_VIP_PKT
#define VI2CSI_VIP_PKT_SIZE 16

// Data
#define VI2CSI_VIP_PKT_VIPDATA                    (15):(0)
#define VI2CSI_VIP_PKT_VIPDATA_ROW                      0

//SENSOR2CIL_TIMING

// Packet SENSOR2CIL_TIMING_PKT
#define SENSOR2CIL_TIMING_PKT_SIZE 73

#define SENSOR2CIL_TIMING_PKT_LPX                 (7):(0)
#define SENSOR2CIL_TIMING_PKT_LPX_ROW                   0

#define SENSOR2CIL_TIMING_PKT_HS_PREPARE                  (15):(8)
#define SENSOR2CIL_TIMING_PKT_HS_PREPARE_ROW                    0

#define SENSOR2CIL_TIMING_PKT_HS_ZERO                     (23):(16)
#define SENSOR2CIL_TIMING_PKT_HS_ZERO_ROW                       0

#define SENSOR2CIL_TIMING_PKT_HS_TRAIL                    (31):(24)
#define SENSOR2CIL_TIMING_PKT_HS_TRAIL_ROW                      0

#define SENSOR2CIL_TIMING_PKT_CLK_ZERO                    (39):(32)
#define SENSOR2CIL_TIMING_PKT_CLK_ZERO_ROW                      0

#define SENSOR2CIL_TIMING_PKT_CLK_PRE                     (47):(40)
#define SENSOR2CIL_TIMING_PKT_CLK_PRE_ROW                       0

#define SENSOR2CIL_TIMING_PKT_CLK_POST                    (55):(48)
#define SENSOR2CIL_TIMING_PKT_CLK_POST_ROW                      0

#define SENSOR2CIL_TIMING_PKT_CLK_TRAIL                   (63):(56)
#define SENSOR2CIL_TIMING_PKT_CLK_TRAIL_ROW                     0

#define SENSOR2CIL_TIMING_PKT_HS_EXIT                     (71):(64)
#define SENSOR2CIL_TIMING_PKT_HS_EXIT_ROW                       0

// default to use RTL internal
#define SENSOR2CIL_TIMING_PKT_RANDOM                      (72):(72)
#define SENSOR2CIL_TIMING_PKT_RANDOM_ROW                        0

//SENSOR2CIL_COMMAND

// Packet SENSOR2CIL_COMMAND_PKT
#define SENSOR2CIL_COMMAND_PKT_SIZE 33

// 
// NO_OP    =0x00000000,   
// ESC_ULPS =0x1, // escape mode: ultra low power state
// ESC_LPDT =0x2, // escape mode: low power data transmission
// ESC_RAR  =0x3, // escape mode: remote application reset
// SOT_ERR  =0x4  // use SOT_CODE for SOT error injection
// FR_HSCLK =0x5  // set high speed clock free running
#define SENSOR2CIL_COMMAND_PKT_CMD                        (4):(0)
#define SENSOR2CIL_COMMAND_PKT_CMD_ROW                  0

// sot or escape delay in esc mode
#define SENSOR2CIL_COMMAND_PKT_PARAM                      (12):(5)
#define SENSOR2CIL_COMMAND_PKT_PARAM_ROW                        0

#define SENSOR2CIL_COMMAND_PKT_ESC_MODE_ENTRY_SEQ                 (20):(13)
#define SENSOR2CIL_COMMAND_PKT_ESC_MODE_ENTRY_SEQ_ROW                   0

#define SENSOR2CIL_COMMAND_PKT_ESC_MODE_CODE                      (28):(21)
#define SENSOR2CIL_COMMAND_PKT_ESC_MODE_CODE_ROW                        0

#define SENSOR2CIL_COMMAND_PKT_ESC_MODE_WIDTH                     (32):(29)
#define SENSOR2CIL_COMMAND_PKT_ESC_MODE_WIDTH_ROW                       0

//Internal packets

// Packet CSI_HEADER
#define LWE24D_CSI_HEADER_SIZE 32

// Data type in packet
#define LWE24D_CSI_HEADER_DATA_TYPE                      (5):(0)
#define LWE24D_CSI_HEADER_DATA_TYPE_ROW                        0

// Virtual channel number
#define LWE24D_CSI_HEADER_VIRTUAL_CHANNEL                        (7):(6)
#define LWE24D_CSI_HEADER_VIRTUAL_CHANNEL_ROW                  0

// Number of bytes in packet payload
#define LWE24D_CSI_HEADER_WORD_COUNT                     (23):(8)
#define LWE24D_CSI_HEADER_WORD_COUNT_ROW                       0

// Error correction code for packet
#define LWE24D_CSI_HEADER_ECC                    (31):(24)
#define LWE24D_CSI_HEADER_ECC_ROW                      0

// Packet CSI_RAISE
#define LWE24D_CSI_RAISE_SIZE 20

#define LWE24D_CSI_RAISE_VECTOR                  (4):(0)
#define LWE24D_CSI_RAISE_VECTOR_ROW                    0

#define LWE24D_CSI_RAISE_COUNT                   (15):(8)
#define LWE24D_CSI_RAISE_COUNT_ROW                     0

#define LWE24D_CSI_RAISE_CHID                    (19):(16)
#define LWE24D_CSI_RAISE_CHID_ROW                      0


// Packet CSI_GENERIC_BYTE
#define LWE24D_CSI_GENERIC_BYTE_SIZE 72

#define LWE24D_CSI_GENERIC_BYTE_BYTE0                    (7):(0)
#define LWE24D_CSI_GENERIC_BYTE_BYTE0_ROW                      0

#define LWE24D_CSI_GENERIC_BYTE_BYTE1                    (15):(8)
#define LWE24D_CSI_GENERIC_BYTE_BYTE1_ROW                      0

#define LWE24D_CSI_GENERIC_BYTE_BYTE2                    (23):(16)
#define LWE24D_CSI_GENERIC_BYTE_BYTE2_ROW                      0

#define LWE24D_CSI_GENERIC_BYTE_BYTE3                    (31):(24)
#define LWE24D_CSI_GENERIC_BYTE_BYTE3_ROW                      0

#define LWE24D_CSI_GENERIC_BYTE_BYTE4                    (39):(32)
#define LWE24D_CSI_GENERIC_BYTE_BYTE4_ROW                      0

#define LWE24D_CSI_GENERIC_BYTE_BYTE5                    (47):(40)
#define LWE24D_CSI_GENERIC_BYTE_BYTE5_ROW                      0

#define LWE24D_CSI_GENERIC_BYTE_BYTE6                    (55):(48)
#define LWE24D_CSI_GENERIC_BYTE_BYTE6_ROW                      0

#define LWE24D_CSI_GENERIC_BYTE_BYTE7                    (63):(56)
#define LWE24D_CSI_GENERIC_BYTE_BYTE7_ROW                      0

#define LWE24D_CSI_GENERIC_BYTE_BYTE8                    (71):(64)
#define LWE24D_CSI_GENERIC_BYTE_BYTE8_ROW                      0


// Packet CSI_RGB_666
#define LWE24D_CSI_RGB_666_SIZE 72

#define LWE24D_CSI_RGB_666_B0                    (5):(0)
#define LWE24D_CSI_RGB_666_B0_ROW                      0

#define LWE24D_CSI_RGB_666_G0                    (11):(6)
#define LWE24D_CSI_RGB_666_G0_ROW                      0

#define LWE24D_CSI_RGB_666_R0                    (17):(12)
#define LWE24D_CSI_RGB_666_R0_ROW                      0

#define LWE24D_CSI_RGB_666_B1                    (23):(18)
#define LWE24D_CSI_RGB_666_B1_ROW                      0

#define LWE24D_CSI_RGB_666_G1                    (29):(24)
#define LWE24D_CSI_RGB_666_G1_ROW                      0

#define LWE24D_CSI_RGB_666_R1                    (35):(30)
#define LWE24D_CSI_RGB_666_R1_ROW                      0

#define LWE24D_CSI_RGB_666_B2                    (41):(36)
#define LWE24D_CSI_RGB_666_B2_ROW                      0

#define LWE24D_CSI_RGB_666_G2                    (47):(42)
#define LWE24D_CSI_RGB_666_G2_ROW                      0

#define LWE24D_CSI_RGB_666_R2                    (53):(48)
#define LWE24D_CSI_RGB_666_R2_ROW                      0

#define LWE24D_CSI_RGB_666_B3                    (59):(54)
#define LWE24D_CSI_RGB_666_B3_ROW                      0

#define LWE24D_CSI_RGB_666_G3                    (65):(60)
#define LWE24D_CSI_RGB_666_G3_ROW                      0

#define LWE24D_CSI_RGB_666_R3                    (71):(66)
#define LWE24D_CSI_RGB_666_R3_ROW                      0


// Packet CSI_RGB_565
#define LWE24D_CSI_RGB_565_SIZE 16

#define LWE24D_CSI_RGB_565_B0                    (4):(0)
#define LWE24D_CSI_RGB_565_B0_ROW                      0

#define LWE24D_CSI_RGB_565_G0                    (10):(5)
#define LWE24D_CSI_RGB_565_G0_ROW                      0

#define LWE24D_CSI_RGB_565_R0                    (15):(11)
#define LWE24D_CSI_RGB_565_R0_ROW                      0


// Packet CSI_RAW_6
#define LWE24D_CSI_RAW_6_SIZE 24

#define LWE24D_CSI_RAW_6_S0                      (5):(0)
#define LWE24D_CSI_RAW_6_S0_ROW                        0

#define LWE24D_CSI_RAW_6_S1                      (11):(6)
#define LWE24D_CSI_RAW_6_S1_ROW                        0

#define LWE24D_CSI_RAW_6_S2                      (17):(12)
#define LWE24D_CSI_RAW_6_S2_ROW                        0

#define LWE24D_CSI_RAW_6_S3                      (23):(18)
#define LWE24D_CSI_RAW_6_S3_ROW                        0


// Packet CSI_RAW_7
#define LWE24D_CSI_RAW_7_SIZE 56

#define LWE24D_CSI_RAW_7_S0                      (6):(0)
#define LWE24D_CSI_RAW_7_S0_ROW                        0

#define LWE24D_CSI_RAW_7_S1                      (13):(7)
#define LWE24D_CSI_RAW_7_S1_ROW                        0

#define LWE24D_CSI_RAW_7_S2                      (20):(14)
#define LWE24D_CSI_RAW_7_S2_ROW                        0

#define LWE24D_CSI_RAW_7_S3                      (27):(21)
#define LWE24D_CSI_RAW_7_S3_ROW                        0

#define LWE24D_CSI_RAW_7_S4                      (34):(28)
#define LWE24D_CSI_RAW_7_S4_ROW                        0

#define LWE24D_CSI_RAW_7_S5                      (41):(35)
#define LWE24D_CSI_RAW_7_S5_ROW                        0

#define LWE24D_CSI_RAW_7_S6                      (48):(42)
#define LWE24D_CSI_RAW_7_S6_ROW                        0

#define LWE24D_CSI_RAW_7_S7                      (55):(49)
#define LWE24D_CSI_RAW_7_S7_ROW                        0


// Packet CSI_RAW_10
#define LWE24D_CSI_RAW_10_SIZE 40

#define LWE24D_CSI_RAW_10_S0                     (7):(0)
#define LWE24D_CSI_RAW_10_S0_ROW                       0

#define LWE24D_CSI_RAW_10_S1                     (15):(8)
#define LWE24D_CSI_RAW_10_S1_ROW                       0

#define LWE24D_CSI_RAW_10_S2                     (23):(16)
#define LWE24D_CSI_RAW_10_S2_ROW                       0

#define LWE24D_CSI_RAW_10_S3                     (31):(24)
#define LWE24D_CSI_RAW_10_S3_ROW                       0

#define LWE24D_CSI_RAW_10_L0                     (33):(32)
#define LWE24D_CSI_RAW_10_L0_ROW                       0

#define LWE24D_CSI_RAW_10_L1                     (35):(34)
#define LWE24D_CSI_RAW_10_L1_ROW                       0

#define LWE24D_CSI_RAW_10_L2                     (37):(36)
#define LWE24D_CSI_RAW_10_L2_ROW                       0

#define LWE24D_CSI_RAW_10_L3                     (39):(38)
#define LWE24D_CSI_RAW_10_L3_ROW                       0


// Packet CSI_RAW_12
#define LWE24D_CSI_RAW_12_SIZE 24

#define LWE24D_CSI_RAW_12_S0                     (7):(0)
#define LWE24D_CSI_RAW_12_S0_ROW                       0

#define LWE24D_CSI_RAW_12_S1                     (15):(8)
#define LWE24D_CSI_RAW_12_S1_ROW                       0

#define LWE24D_CSI_RAW_12_L0                     (19):(16)
#define LWE24D_CSI_RAW_12_L0_ROW                       0

#define LWE24D_CSI_RAW_12_L1                     (23):(20)
#define LWE24D_CSI_RAW_12_L1_ROW                       0


// Packet CSI_RAW_14
#define LWE24D_CSI_RAW_14_SIZE 56

#define LWE24D_CSI_RAW_14_S0                     (7):(0)
#define LWE24D_CSI_RAW_14_S0_ROW                       0

#define LWE24D_CSI_RAW_14_S1                     (15):(8)
#define LWE24D_CSI_RAW_14_S1_ROW                       0

#define LWE24D_CSI_RAW_14_S2                     (23):(16)
#define LWE24D_CSI_RAW_14_S2_ROW                       0

#define LWE24D_CSI_RAW_14_S3                     (31):(24)
#define LWE24D_CSI_RAW_14_S3_ROW                       0

#define LWE24D_CSI_RAW_14_L0                     (37):(32)
#define LWE24D_CSI_RAW_14_L0_ROW                       0

#define LWE24D_CSI_RAW_14_L1                     (43):(38)
#define LWE24D_CSI_RAW_14_L1_ROW                       0

#define LWE24D_CSI_RAW_14_L2                     (49):(44)
#define LWE24D_CSI_RAW_14_L2_ROW                       0

#define LWE24D_CSI_RAW_14_L3                     (55):(50)
#define LWE24D_CSI_RAW_14_L3_ROW                       0

//defines, used by CSI CModel
//CSI-2 Data Types
// SSP = Synchronization Short Packet
#define LWE24D_CSI_DT_SSP_FS   0
// Frame Start
#define LWE24D_CSI_DT_SSP_FE   1
// Frame End
#define LWE24D_CSI_DT_SSP_LS   2
// Line Start
#define LWE24D_CSI_DT_SSP_LE   3
// Line End
#define LWE24D_CSI_DT_SSP_R1   4
// Reserved 1
#define LWE24D_CSI_DT_SSP_R2   5
// Reserved 2
#define LWE24D_CSI_DT_SSP_R3   6
// Reserved 3
#define LWE24D_CSI_DT_SSP_R4   7
// Reserved 4
// GSP = Generic Short Packet
#define LWE24D_CSI_DT_GSP_G1   8
// Generic Short Packet Code 1
#define LWE24D_CSI_DT_GSP_G2   9
// Generic Short Packet Code 2
#define LWE24D_CSI_DT_GSP_G3   10
// Generic Short Packet Code 3
#define LWE24D_CSI_DT_GSP_G4   11
// Generic Short Packet Code 4
#define LWE24D_CSI_DT_GSP_G5   12
// Generic Short Packet Code 5
#define LWE24D_CSI_DT_GSP_G6   13
// Generic Short Packet Code 6
#define LWE24D_CSI_DT_GSP_G7   14
// Generic Short Packet Code 7
#define LWE24D_CSI_DT_GSP_G8   15
// Generic Short Packet Code 8
// GED = Generic 8-bit Data
#define LWE24D_CSI_DT_GED_NULL 16
// Null 
#define LWE24D_CSI_DT_GED_BLANK        17
// Blanking Data 
#define LWE24D_CSI_DT_GED_ED   18
// Embedded 8-bit non Image Data
#define LWE24D_CSI_DT_GED_R1   19
// Reserved
#define LWE24D_CSI_DT_GED_R2   20
// Reserved
#define LWE24D_CSI_DT_GED_R3   21
// Reserved
#define LWE24D_CSI_DT_GED_R4   22
// Reserved
#define LWE24D_CSI_DT_GED_R5   23
// Reserved
// YUV = YUV Image Data Types
#define LWE24D_CSI_DT_YUV_420_8        24
// YUV420 8-bit
#define LWE24D_CSI_DT_YUV_420_10       25
// YUV420 10-bit
#define LWE24D_CSI_DT_YUV_420_L_8      26
// Legacy YUV420 8-bit
#define LWE24D_CSI_DT_YUV_R1   27
// Reserved
#define LWE24D_CSI_DT_YUV_420_CSPS_8   28
// YUV420 8-bit (Chroma Shifted Pixel Sampling)
#define LWE24D_CSI_DT_YUV_420_CSPS_10  29
// YUV420 10-bit (Chroma Shifted Pixel Sampling)
#define LWE24D_CSI_DT_YUV_422_8        30
// YUV422 8-bit
#define LWE24D_CSI_DT_YUV_422_10       31
// YUV422 10-bit
// RGB = RGB Image Data Types
#define LWE24D_CSI_DT_RGB_444  32
// RGB444
#define LWE24D_CSI_DT_RGB_555  33
// RGB555
#define LWE24D_CSI_DT_RGB_565  34
// RGB565
#define LWE24D_CSI_DT_RGB_666  35
// RGB666
#define LWE24D_CSI_DT_RGB_888  36
// RGB888
#define LWE24D_CSI_DT_RGB_R1   37
// Reserved
#define LWE24D_CSI_DT_RGB_R2   38
// Reserved
#define LWE24D_CSI_DT_RGB_R3   39
// Reserved
// RAW Image Data Types
#define LWE24D_CSI_DT_RAW_6    40
// RAW6
#define LWE24D_CSI_DT_RAW_7    41
// RAW7
#define LWE24D_CSI_DT_RAW_8    42
// RAW8
#define LWE24D_CSI_DT_RAW_10   43
// RAW10
#define LWE24D_CSI_DT_RAW_12   44
// RAW12
#define LWE24D_CSI_DT_RAW_14   45
// RAW14
#define LWE24D_CSI_DT_RAW_R1   46
// Reserved
#define LWE24D_CSI_DT_RAW_R2   47
// Reserved
// UED = User Defined 8-bit Data
#define LWE24D_CSI_DT_UED_U1   48
// User Defined 8-bit Data Type 1
#define LWE24D_CSI_DT_UED_U2   49
// User Defined 8-bit Data Type 2
#define LWE24D_CSI_DT_UED_U3   50
// User Defined 8-bit Data Type 3
#define LWE24D_CSI_DT_UED_U4   51
// User Defined 8-bit Data Type 4
#define LWE24D_CSI_DT_UED_R1   52
// Reserved
#define LWE24D_CSI_DT_UED_R2   53
// Reserved
#define LWE24D_CSI_DT_UED_R3   54
// Reserved
#define LWE24D_CSI_DT_UED_R4   55

/*
//
// REGISTER LIST
//
#define LIST_ARVI_REGS(_op_) \
_op_(LWE24D_VI_OUT_1_INCR_SYNCPT_0) \
_op_(LWE24D_VI_OUT_1_INCR_SYNCPT_CNTRL_0) \
_op_(LWE24D_VI_OUT_1_INCR_SYNCPT_ERROR_0) \
_op_(LWE24D_VI_OUT_2_INCR_SYNCPT_0) \
_op_(LWE24D_VI_OUT_2_INCR_SYNCPT_CNTRL_0) \
_op_(LWE24D_VI_OUT_2_INCR_SYNCPT_ERROR_0) \
_op_(LWE24D_VI_MISC_INCR_SYNCPT_0) \
_op_(LWE24D_VI_MISC_INCR_SYNCPT_CNTRL_0) \
_op_(LWE24D_VI_MISC_INCR_SYNCPT_ERROR_0) \
_op_(LWE24D_VI_CONT_SYNCPT_OUT_1_0) \
_op_(LWE24D_VI_CONT_SYNCPT_OUT_2_0) \
_op_(LWE24D_VI_CONT_SYNCPT_VIP_VSYNC_0) \
_op_(LWE24D_VI_CONT_SYNCPT_VI2EPP_0) \
_op_(LWE24D_VI_CONT_SYNCPT_CSI_PPA_FRAME_START_0) \
_op_(LWE24D_VI_CONT_SYNCPT_CSI_PPA_FRAME_END_0) \
_op_(LWE24D_VI_CONT_SYNCPT_CSI_PPB_FRAME_START_0) \
_op_(LWE24D_VI_CONT_SYNCPT_CSI_PPB_FRAME_END_0) \
_op_(LWE24D_VI_CTXSW_0) \
_op_(LWE24D_VI_INTSTATUS_0) \
_op_(LWE24D_VI_VI_INPUT_CONTROL_0) \
_op_(LWE24D_VI_VI_CORE_CONTROL_0) \
_op_(LWE24D_VI_VI_FIRST_OUTPUT_CONTROL_0) \
_op_(LWE24D_VI_VI_SECOND_OUTPUT_CONTROL_0) \
_op_(LWE24D_VI_HOST_INPUT_FRAME_SIZE_0) \
_op_(LWE24D_VI_HOST_H_ACTIVE_0) \
_op_(LWE24D_VI_HOST_V_ACTIVE_0) \
_op_(LWE24D_VI_VIP_H_ACTIVE_0) \
_op_(LWE24D_VI_VIP_V_ACTIVE_0) \
_op_(LWE24D_VI_VI_PEER_CONTROL_0) \
_op_(LWE24D_VI_VI_DMA_SELECT_0) \
_op_(LWE24D_VI_HOST_DMA_WRITE_BUFFER_0) \
_op_(LWE24D_VI_HOST_DMA_BASE_ADDRESS_0) \
_op_(LWE24D_VI_HOST_DMA_WRITE_BUFFER_STATUS_0) \
_op_(LWE24D_VI_HOST_DMA_WRITE_PEND_BUFCOUNT_0) \
_op_(LWE24D_VI_VB0_START_ADDRESS_FIRST_0) \
_op_(LWE24D_VI_VB0_BASE_ADDRESS_FIRST_0) \
_op_(LWE24D_VI_VB0_START_ADDRESS_U_0) \
_op_(LWE24D_VI_VB0_BASE_ADDRESS_U_0) \
_op_(LWE24D_VI_VB0_START_ADDRESS_V_0) \
_op_(LWE24D_VI_VB0_BASE_ADDRESS_V_0) \
_op_(LWE24D_VI_VB0_SCRATCH_ADDRESS_UV_0) \
_op_(LWE24D_VI_FIRST_OUTPUT_FRAME_SIZE_0) \
_op_(LWE24D_VI_VB0_COUNT_FIRST_0) \
_op_(LWE24D_VI_VB0_SIZE_FIRST_0) \
_op_(LWE24D_VI_VB0_BUFFER_STRIDE_FIRST_0) \
_op_(LWE24D_VI_VB0_START_ADDRESS_SECOND_0) \
_op_(LWE24D_VI_VB0_BASE_ADDRESS_SECOND_0) \
_op_(LWE24D_VI_SECOND_OUTPUT_FRAME_SIZE_0) \
_op_(LWE24D_VI_VB0_COUNT_SECOND_0) \
_op_(LWE24D_VI_VB0_SIZE_SECOND_0) \
_op_(LWE24D_VI_VB0_BUFFER_STRIDE_SECOND_0) \
_op_(LWE24D_VI_H_LPF_CONTROL_0) \
_op_(LWE24D_VI_H_DOWNSCALE_CONTROL_0) \
_op_(LWE24D_VI_V_DOWNSCALE_CONTROL_0) \
_op_(LWE24D_VI_CSC_Y_0) \
_op_(LWE24D_VI_CSC_UV_R_0) \
_op_(LWE24D_VI_CSC_UV_G_0) \
_op_(LWE24D_VI_CSC_UV_B_0) \
_op_(LWE24D_VI_CSC_ALPHA_0) \
_op_(LWE24D_VI_HOST_VSYNC_0) \
_op_(LWE24D_VI_COMMAND_0) \
_op_(LWE24D_VI_HOST_FIFO_STATUS_0) \
_op_(LWE24D_VI_INTERRUPT_MASK_0) \
_op_(LWE24D_VI_INTERRUPT_TYPE_SELECT_0) \
_op_(LWE24D_VI_INTERRUPT_POLARITY_SELECT_0) \
_op_(LWE24D_VI_INTERRUPT_STATUS_0) \
_op_(LWE24D_VI_VIP_INPUT_STATUS_0) \
_op_(LWE24D_VI_VIDEO_BUFFER_STATUS_0) \
_op_(LWE24D_VI_SYNC_OUTPUT_0) \
_op_(LWE24D_VI_VVS_OUTPUT_DELAY_0) \
_op_(LWE24D_VI_PWM_CONTROL_0) \
_op_(LWE24D_VI_PWM_SELECT_PULSE_A_0) \
_op_(LWE24D_VI_PWM_SELECT_PULSE_B_0) \
_op_(LWE24D_VI_PWM_SELECT_PULSE_C_0) \
_op_(LWE24D_VI_PWM_SELECT_PULSE_D_0) \
_op_(LWE24D_VI_VI_DATA_INPUT_CONTROL_0) \
_op_(LWE24D_VI_PIN_INPUT_ENABLE_0) \
_op_(LWE24D_VI_PIN_OUTPUT_ENABLE_0) \
_op_(LWE24D_VI_PIN_ILWERSION_0) \
_op_(LWE24D_VI_PIN_INPUT_DATA_0) \
_op_(LWE24D_VI_PIN_OUTPUT_DATA_0) \
_op_(LWE24D_VI_PIN_OUTPUT_SELECT_0) \
_op_(LWE24D_VI_RAISE_VIP_BUFFER_FIRST_OUTPUT_0) \
_op_(LWE24D_VI_RAISE_VIP_FRAME_FIRST_OUTPUT_0) \
_op_(LWE24D_VI_RAISE_VIP_BUFFER_SECOND_OUTPUT_0) \
_op_(LWE24D_VI_RAISE_VIP_FRAME_SECOND_OUTPUT_0) \
_op_(LWE24D_VI_RAISE_HOST_FIRST_OUTPUT_0) \
_op_(LWE24D_VI_RAISE_HOST_SECOND_OUTPUT_0) \
_op_(LWE24D_VI_RAISE_EPP_0) \
_op_(LWE24D_VI_CAMERA_CONTROL_0) \
_op_(LWE24D_VI_VI_ENABLE_0) \
_op_(LWE24D_VI_VI_ENABLE_2_0) \
_op_(LWE24D_VI_VI_RAISE_0) \
_op_(LWE24D_VI_Y_FIFO_WRITE_0) \
_op_(LWE24D_VI_U_FIFO_WRITE_0) \
_op_(LWE24D_VI_V_FIFO_WRITE_0) \
_op_(LWE24D_VI_VI_MCCIF_FIFOCTRL_0) \
_op_(LWE24D_VI_TIMEOUT_WCOAL_VI_0) \
_op_(LWE24D_VI_MCCIF_VIRUV_HP_0) \
_op_(LWE24D_VI_MCCIF_VIWSB_HP_0) \
_op_(LWE24D_VI_MCCIF_VIWU_HP_0) \
_op_(LWE24D_VI_MCCIF_VIWV_HP_0) \
_op_(LWE24D_VI_MCCIF_VIWY_HP_0) \
_op_(LWE24D_VI_CSI_PPA_RAISE_FRAME_START_0) \
_op_(LWE24D_VI_CSI_PPA_RAISE_FRAME_END_0) \
_op_(LWE24D_VI_CSI_PPB_RAISE_FRAME_START_0) \
_op_(LWE24D_VI_CSI_PPB_RAISE_FRAME_END_0) \
_op_(LWE24D_VI_CSI_PPA_H_ACTIVE_0) \
_op_(LWE24D_VI_CSI_PPA_V_ACTIVE_0) \
_op_(LWE24D_VI_CSI_PPB_H_ACTIVE_0) \
_op_(LWE24D_VI_CSI_PPB_V_ACTIVE_0) \
_op_(LWE24D_VI_ISP_H_ACTIVE_0) \
_op_(LWE24D_VI_ISP_V_ACTIVE_0) \
_op_(LWE24D_VI_STREAM_1_RESOURCE_DEFINE_0) \
_op_(LWE24D_VI_STREAM_2_RESOURCE_DEFINE_0) \
_op_(LWE24D_VI_RAISE_STREAM_1_DONE_0) \
_op_(LWE24D_VI_RAISE_STREAM_2_DONE_0) \
_op_(LWE24D_VI_TS_MODE_0) \
_op_(LWE24D_VI_TS_CONTROL_0) \
_op_(LWE24D_VI_TS_PACKET_COUNT_0) \
_op_(LWE24D_VI_TS_ERROR_COUNT_0) \
_op_(LWE24D_VI_TS_CPU_FLOW_CTL_0) \
_op_(LWE24D_VI_VB0_CHROMA_BUFFER_STRIDE_FIRST_0) \
_op_(LWE24D_VI_VB0_CHROMA_LINE_STRIDE_FIRST_0) \
_op_(LWE24D_VI_EPP_LINES_PER_BUFFER_0) \
_op_(LWE24D_VI_BUFFER_RELEASE_OUTPUT1_0) \
_op_(LWE24D_VI_BUFFER_RELEASE_OUTPUT2_0) \
_op_(LWE24D_VI_DEBUG_FLOW_CONTROL_COUNTER_OUTPUT1_0) \
_op_(LWE24D_VI_DEBUG_FLOW_CONTROL_COUNTER_OUTPUT2_0) \
_op_(LWE24D_VI_TERMINATE_BW_FIRST_0) \
_op_(LWE24D_VI_TERMINATE_BW_SECOND_0) \
_op_(LWE24D_VI_VB0_FIRST_BUFFER_ADDR_MODE_0) \
_op_(LWE24D_VI_VB0_SECOND_BUFFER_ADDR_MODE_0) \
_op_(LWE24D_VI_RESERVE_0_0) \
_op_(LWE24D_VI_RESERVE_1_0) \
_op_(LWE24D_VI_RESERVE_2_0) \
_op_(LWE24D_VI_RESERVE_3_0) \
_op_(LWE24D_VI_RESERVE_4_0) \
_op_(LWE24D_VI_MCCIF_VIRUV_HYST_0) \
_op_(LWE24D_VI_MCCIF_VIWSB_HYST_0) \
_op_(LWE24D_VI_MCCIF_VIWU_HYST_0) \
_op_(LWE24D_VI_MCCIF_VIWV_HYST_0) \
_op_(LWE24D_VI_MCCIF_VIWY_HYST_0) \
_op_(LWE24D_CSI_VI_INPUT_STREAM_CONTROL_0) \
_op_(LWE24D_CSI_HOST_INPUT_STREAM_CONTROL_0) \
_op_(LWE24D_CSI_INPUT_STREAM_A_CONTROL_0) \
_op_(LWE24D_CSI_PIXEL_STREAM_A_CONTROL0_0) \
_op_(LWE24D_CSI_PIXEL_STREAM_A_CONTROL1_0) \
_op_(LWE24D_CSI_PIXEL_STREAM_A_WORD_COUNT_0) \
_op_(LWE24D_CSI_PIXEL_STREAM_A_GAP_0) \
_op_(LWE24D_CSI_PIXEL_STREAM_PPA_COMMAND_0) \
_op_(LWE24D_CSI_INPUT_STREAM_B_CONTROL_0) \
_op_(LWE24D_CSI_PIXEL_STREAM_B_CONTROL0_0) \
_op_(LWE24D_CSI_PIXEL_STREAM_B_CONTROL1_0) \
_op_(LWE24D_CSI_PIXEL_STREAM_B_WORD_COUNT_0) \
_op_(LWE24D_CSI_PIXEL_STREAM_B_GAP_0) \
_op_(LWE24D_CSI_PIXEL_STREAM_PPB_COMMAND_0) \
_op_(LWE24D_CSI_PHY_CIL_COMMAND_0) \
_op_(LWE24D_CSI_PHY_CILA_CONTROL0_0) \
_op_(LWE24D_CSI_PHY_CILB_CONTROL0_0) \
_op_(LWE24D_CSI_CSI_PIXEL_PARSER_STATUS_0) \
_op_(LWE24D_CSI_CSI_CIL_STATUS_0) \
_op_(LWE24D_CSI_CSI_PIXEL_PARSER_INTERRUPT_MASK_0) \
_op_(LWE24D_CSI_CSI_CIL_INTERRUPT_MASK_0) \
_op_(LWE24D_CSI_CSI_READONLY_STATUS_0) \
_op_(LWE24D_CSI_ESCAPE_MODE_COMMAND_0) \
_op_(LWE24D_CSI_ESCAPE_MODE_DATA_0) \
_op_(LWE24D_CSI_CILA_PAD_CONFIG0_0) \
_op_(LWE24D_CSI_CILA_PAD_CONFIG1_0) \
_op_(LWE24D_CSI_CILB_PAD_CONFIG0_0) \
_op_(LWE24D_CSI_CILB_PAD_CONFIG1_0) \
_op_(LWE24D_CSI_CIL_PAD_CONFIG0_0) \
_op_(LWE24D_CSI_CILA_MIPI_CAL_CONFIG_0) \
_op_(LWE24D_CSI_CILB_MIPI_CAL_CONFIG_0) \
_op_(LWE24D_CSI_CIL_MIPI_CAL_STATUS_0) \
_op_(LWE24D_CSI_CLKEN_OVERRIDE_0) \
_op_(LWE24D_CSI_DEBUG_CONTROL_0) \
_op_(LWE24D_CSI_DEBUG_COUNTER_0_0) \
_op_(LWE24D_CSI_DEBUG_COUNTER_1_0) \
_op_(LWE24D_CSI_DEBUG_COUNTER_2_0) \
_op_(LWE24D_CSI_PIXEL_STREAM_A_EXPECTED_FRAME_0) \
_op_(LWE24D_CSI_PIXEL_STREAM_B_EXPECTED_FRAME_0) \
_op_(LWE24D_CSI_DSI_MIPI_CAL_CONFIG_0)

//
// ADDRESS SPACES
//
#define BASE_ADDRESS_VI         0x000000000000000
#define BASE_ADDRESS_CSI        0x000000000000200

//
// ARVI REGISTER BANKS
//
#define VI0_FIRST_REG 0x00000000000 // VI_OUT_1_INCR_SYNCPT_0
#define VI0_LAST_REG 0x00000000002 // VI_OUT_1_INCR_SYNCPT_ERROR_0
#define VI1_FIRST_REG 0x00000000008 // VI_OUT_2_INCR_SYNCPT_0
#define VI1_LAST_REG 0x0000000000a // VI_OUT_2_INCR_SYNCPT_ERROR_0
#define VI2_FIRST_REG 0x00000000010 // VI_MISC_INCR_SYNCPT_0
#define VI2_LAST_REG 0x00000000012 // VI_MISC_INCR_SYNCPT_ERROR_0
#define VI3_FIRST_REG 0x00000000018 // VI_CONT_SYNCPT_OUT_1_0
#define VI3_LAST_REG 0x0000000009d // VI_MCCIF_VIWY_HYST_0
#define CSI0_FIRST_REG 0x00000000200 // CSI_VI_INPUT_STREAM_CONTROL_0
#define CSI0_LAST_REG 0x00000000200 // CSI_VI_INPUT_STREAM_CONTROL_0
#define CSI1_FIRST_REG 0x00000000202 // CSI_HOST_INPUT_STREAM_CONTROL_0
#define CSI1_LAST_REG 0x00000000202 // CSI_HOST_INPUT_STREAM_CONTROL_0
#define CSI2_FIRST_REG 0x00000000204 // CSI_INPUT_STREAM_A_CONTROL_0
#define CSI2_LAST_REG 0x00000000204 // CSI_INPUT_STREAM_A_CONTROL_0
#define CSI3_FIRST_REG 0x00000000206 // CSI_PIXEL_STREAM_A_CONTROL0_0
#define CSI3_LAST_REG 0x0000000020a // CSI_PIXEL_STREAM_PPA_COMMAND_0
#define CSI4_FIRST_REG 0x0000000020f // CSI_INPUT_STREAM_B_CONTROL_0
#define CSI4_LAST_REG 0x0000000020f // CSI_INPUT_STREAM_B_CONTROL_0
#define CSI5_FIRST_REG 0x00000000211 // CSI_PIXEL_STREAM_B_CONTROL0_0
#define CSI5_LAST_REG 0x00000000215 // CSI_PIXEL_STREAM_PPB_COMMAND_0
#define CSI6_FIRST_REG 0x0000000021a // CSI_PHY_CIL_COMMAND_0
#define CSI6_LAST_REG 0x0000000021c // CSI_PHY_CILB_CONTROL0_0
#define CSI7_FIRST_REG 0x0000000021e // CSI_CSI_PIXEL_PARSER_STATUS_0
#define CSI7_LAST_REG 0x00000000234 // CSI_DSI_MIPI_CAL_CONFIG_0
*/

#endif // ifndef _cl_e24d_h_
